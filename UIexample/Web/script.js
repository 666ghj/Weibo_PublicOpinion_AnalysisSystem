"use strict";
let mapping = {};
const dbStorage = {
  dbName: "vvSearchCache",
  dbVersion: 1,
  async init() {
    if (this._dbPromise) return this._dbPromise;
    this._dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      request.onerror = (event) => {
        reject(event.target.error);
      };
      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("indices"))
          db.createObjectStore("indices", { keyPath: "key" });
        if (!db.objectStoreNames.contains("mappings"))
          db.createObjectStore("mappings", { keyPath: "key" });
        if (!db.objectStoreNames.contains("databases"))
          db.createObjectStore("databases", { keyPath: "key" });
      };
    });
    return this._dbPromise;
  },
  async getItem(storeName, key) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(storeName, "readonly");
        const store = transaction.objectStore(storeName);
        const request = store.get(key);
        request.onsuccess = () => {
          resolve(request.result ? request.result.value : null);
        };
        request.onerror = (event) => {
          reject(event.target.error);
        };
      });
    } catch (error) {
      return null;
    }
  },
  async setItem(storeName, key, value) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(storeName, "readwrite");
        const store = transaction.objectStore(storeName);
        const request = store.put({ key, value });
        request.onsuccess = () => {
          resolve();
        };
        request.onerror = (event) => {
          reject(event.target.error);
        };
      });
    } catch (error) {
      throw error;
    }
  },
  async removeItem(storeName, key) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(storeName, "readwrite");
        const store = transaction.objectStore(storeName);
        const request = store.delete(key);
        request.onsuccess = () => {
          resolve();
        };
        request.onerror = (event) => {
          reject(event.target.error);
        };
      });
    } catch (error) {
      throw error;
    }
  },
  async getAllKeys(storeName) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(storeName, "readonly");
        const store = transaction.objectStore(storeName);
        const request = store.getAllKeys();
        request.onsuccess = () => {
          resolve(request.result);
        };
        request.onerror = (event) => {
          reject(event.target.error);
        };
      });
    } catch (error) {
      return [];
    }
  },
};
const RequestController = {
  queue: new Map(),
  maxConcurrent: 4,

  async enqueue(key, requestFn) {
    if (this.queue.has(key)) {
      return this.queue.get(key);
    }
    while (this.queue.size >= this.maxConcurrent) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    const promise = requestFn().finally(() => {
      this.queue.delete(key);
    });

    this.queue.set(key, promise);
    return promise;
  },
};
const indexCache = {
  data: new Map(),
  preloadQueue: new Set(),
  preloadPromises: new Map(),

  async get(groupIndex, baseDir) {
    const cacheKey = `${baseDir}/${groupIndex}`;

    if (this.data.has(cacheKey)) {
      return this.data.get(cacheKey);
    }

    if (this.preloadPromises.has(cacheKey)) {
      return this.preloadPromises.get(cacheKey);
    }

    return RequestController.enqueue(cacheKey, async () => {
      try {
        if (this.data.has(cacheKey)) {
          return this.data.get(cacheKey);
        }

        const cachedData = await dbStorage.getItem("indices", cacheKey);
        if (cachedData) {
          const arrayBuffer = this._base64ToArrayBuffer(cachedData);
          this.data.set(cacheKey, arrayBuffer);
          return arrayBuffer;
        }

        const indexData = await this._fetchIndex(groupIndex, baseDir);
        this.data.set(cacheKey, indexData);

        this._saveToCache(cacheKey, indexData).catch(() => {});

        return indexData;
      } catch (error) {
        console.error(`Failed to load index ${cacheKey}:`, error);
        throw error;
      }
    });
  },

  async _saveToCache(cacheKey, data) {
    const base64Data = this._arrayBufferToBase64(data);
    await dbStorage.setItem("indices", cacheKey, base64Data);
  },

  async preload(groupIndex, baseDir) {
    const cacheKey = `${baseDir}/${groupIndex}`;
    if (this.data.has(cacheKey) || this.preloadQueue.has(cacheKey)) {
      return;
    }

    this.preloadQueue.add(cacheKey);
    const promise = this.get(groupIndex, baseDir)
      .catch(() => {})
      .finally(() => {
        this.preloadQueue.delete(cacheKey);
        this.preloadPromises.delete(cacheKey);
      });

    this.preloadPromises.set(cacheKey, promise);
  },

  async _fetchIndex(groupIndex, baseDir) {
    const cacheKey = `${baseDir}/${groupIndex}`;

    const indexUrl = `${baseDir}/${groupIndex}.index`;

    try {
      const indexResponse = await fetch(indexUrl, {
        method: "GET",
        mode: "cors",
        credentials: "omit",
        cache: "no-cache",
        headers: {
          Accept: "application/octet-stream",
        },
        referrerPolicy: "no-referrer",
      });

      if (!indexResponse.ok) {
        throw new Error(
          `Failed to fetch index: ${indexResponse.status} ${indexResponse.statusText}`,
        );
      }

      const headers = Object.fromEntries(indexResponse.headers.entries());
      const contentType = headers["content-type"];

      const compressedData = await indexResponse.arrayBuffer();

      if (compressedData.byteLength === 0) {
        throw new Error("Received empty response");
      }

      const header = new Uint8Array(compressedData.slice(0, 2));

      let decompressedData;

      if (header[0] === 0x1f && header[1] === 0x8b) {
        try {
          const ds = new DecompressionStream("gzip");
          const decompressedStream = new Response(
            compressedData,
          ).body.pipeThrough(ds);
          decompressedData = await new Response(
            decompressedStream,
          ).arrayBuffer();
        } catch (error) {
          console.error("Decompression failed:", error);
          throw error;
        }
      } else {
        decompressedData = compressedData;
      }

      if (decompressedData.byteLength < 16) {
        throw new Error("Data too small");
      }

      const view = new DataView(decompressedData);
      const gridW = view.getUint32(0, true);
      const gridH = view.getUint32(4, true);
      const folderCount = view.getUint32(8, true);

      if (gridW === 0 || gridH === 0 || folderCount === 0) {
        throw new Error("Invalid index format");
      }

      return decompressedData;
    } catch (error) {
      console.error(`Failed to fetch or process index ${indexUrl}:`, error);
      throw error;
    }
  },

  _arrayBufferToBase64(buffer) {
    const binary = [];
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++)
      binary.push(String.fromCharCode(bytes[i]));
    return btoa(binary.join(""));
  },

  _base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
    return bytes.buffer;
  },

  _cleanupLocalStorage() {
    try {
      const cacheKeys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith("indexCache_")) cacheKeys.push(key);
      }
      if (cacheKeys.length > 20)
        for (let i = 0; i < 5; i++) localStorage.removeItem(cacheKeys[i]);
    } catch (e) {}
  },
};
const watermarkImage = new Image();
watermarkImage.src = "watermark.png";
let watermarkLoaded = false;
watermarkImage.onload = () => {
  watermarkLoaded = true;
};
async function extractFrame(folderId, frameNum, baseDir = "") {
  const groupIndex = Math.floor((folderId - 1) / 10);
  const requestKey = `${baseDir}/${groupIndex}/${folderId}/${frameNum}`;

  return RequestController.enqueue(requestKey, async () => {
    try {
      if (!baseDir) {
        throw new Error("Base directory is required");
      }

      const indexData = await indexCache.get(groupIndex, baseDir);

      const dataView = new DataView(indexData);
      let offset = 0;

      const gridW = dataView.getUint32(offset, true);
      offset += 4;
      const gridH = dataView.getUint32(offset, true);
      offset += 4;
      const folderCount = dataView.getUint32(offset, true);
      offset += 4;
      offset += folderCount * 4;
      const fileCount = dataView.getUint32(offset, true);
      offset += 4;

      let left = 0;
      let right = fileCount - 1;
      let startOffset = null;
      let endOffset = null;

      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const recordOffset = offset + mid * 16;
        const currFolder = dataView.getUint32(recordOffset, true);
        const currFrame = dataView.getUint32(recordOffset + 4, true);
        const currFileOffset = Number(
          dataView.getBigUint64(recordOffset + 8, true),
        );

        if (currFolder === folderId && currFrame === frameNum) {
          startOffset = currFileOffset;
          if (mid < fileCount - 1) {
            endOffset = Number(dataView.getBigUint64(recordOffset + 24, true));
          }
          break;
        } else if (
          currFolder < folderId ||
          (currFolder === folderId && currFrame < frameNum)
        ) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }

      if (startOffset === null) {
        throw new Error(`Frame ${frameNum} not found in folder ${folderId}`);
      }

      const imageUrl = `${baseDir}/${groupIndex}.webp`;

      const response = await fetch(imageUrl, {
        method: "GET",
        headers: {
          Range: `bytes=${startOffset}-${endOffset ? endOffset - 1 : ""}`,
        },
        mode: "cors",
        credentials: "omit",
        referrerPolicy: "no-referrer",
      });

      if (response.status === 416 || !response.ok) {
        const fullResponse = await fetch(imageUrl, {
          method: "GET",
          mode: "cors",
          credentials: "omit",
          referrerPolicy: "no-referrer",
        });

        if (!fullResponse.ok) {
          throw new Error(
            `HTTP error: ${fullResponse.status} ${fullResponse.statusText}`,
          );
        }

        return new Blob([await fullResponse.blob()], { type: "image/webp" });
      }

      const data = await response.blob();
      if (!data || data.size === 0) {
        throw new Error("Empty response");
      }

      return new Blob([data], { type: "image/webp" });
    } catch (error) {
      console.error(
        `Error extracting frame ${frameNum} from folder ${folderId}:`,
        error,
      );
      throw new Error(`Failed to load preview image: ${error.message}`);
    }
  });
}
async function loadMapping() {
  try {
    const cachedMapping = await dbStorage.getItem("mappings", "mapping");
    if (cachedMapping) return cachedMapping;
    try {
      const localMapping = localStorage.getItem("mapping");
      if (localMapping) {
        const mappingData = JSON.parse(localMapping);
        try {
          await dbStorage.setItem("mappings", "mapping", mappingData);
          localStorage.removeItem("mapping");
        } catch (e) {}
        return mappingData;
      }
    } catch (e) {}
    const response = await fetch("./mapping.json", {
      referrerPolicy: "no-referrer",
      mode: "cors",
      credentials: "omit",
    });
    if (!response.ok) throw new Error("Failed to load mapping.json");
    const mappingData = await response.json();
    try {
      await dbStorage.setItem("mappings", "mapping", mappingData);
    } catch (e) {
      try {
        localStorage.setItem("mapping", JSON.stringify(mappingData));
      } catch (e) {}
    }
    return mappingData;
  } catch (error) {
    return {};
  }
}
const AppState = {
  isSearching: false,
  randomStringDisplayed: false,
  searchResults: [],
  currentPage: 1,
  itemsPerPage: 20,
  hasMoreResults: true,
  cachedResults: [],
  displayedCount: 0,
  showWatermark: true,
  dbLoaded: false,
  dbLoading: false,
};
const CONFIG = {
  randomStrings: [
    "\u63a2\u7d22VV\u7684\u5f00\u6e90\u4e16\u754c",
    "\u4e3a\u4e1c\u5927\u52a9\u529b",
    "\u641c\u7d22\u4f60\u60f3\u8981\u7684\u5185\u5bb9",
  ],
  apiBaseUrl: "https://vvapi.cicada000.work",
  semanticApiUrl: "https://vvapi.cicada000.work",
  imageBaseUrl: "https://vv.noxylva.org",
  watermarkPath: "watermark.png",
  indexPreloadCount: 26
};
class UIController {
  static updateSearchFormPosition(isSearching) {
    const searchForm = document.getElementById("searchForm");
    const randomStringDisplay = document.getElementById("randomStringDisplay");
    if (isSearching) {
      searchForm.classList.add("searching");
      if (!AppState.randomStringDisplayed) this.showRandomString();
    } else {
      searchForm.classList.remove("searching");
      this.clearRandomString();
    }
  }
  static showRandomString() {
    if (!AppState.randomStringDisplayed) {
      const randomStringDisplay = document.getElementById(
        "randomStringDisplay",
      );
      const randomIndex = Math.floor(
        Math.random() * CONFIG.randomStrings.length,
      );
      randomStringDisplay.textContent = CONFIG.randomStrings[randomIndex];
      AppState.randomStringDisplayed = true;
      randomStringDisplay.classList.remove("fade-out");
      randomStringDisplay.classList.add("fade-in");
    }
  }
  static clearRandomString() {
    const randomStringDisplay = document.getElementById("randomStringDisplay");
    randomStringDisplay.classList.remove("fade-in");
    randomStringDisplay.classList.add("fade-out");
    setTimeout(() => {
      randomStringDisplay.textContent = "";
      AppState.randomStringDisplayed = false;
    }, 300);
  }
}
class SearchController {
  static validateSearchInput(query) {
    return query && query.trim().length > 0;
  }

  static async performSearch(query, minRatio, minSimilarity) {
    const isSemanticSearch = document
      .getElementById("semanticToggle")
      .classList.contains("active");
    if (!isSemanticSearch) {
      if (window.subtitleDB && window.subtitleDB.isLoaded) {
        try {
          const localResults = await window.subtitleDB.search(
            query,
            minRatio,
            minSimilarity,
          );

          if (localResults && Array.isArray(localResults)) {
            return {
              status: "success",
              data: localResults,
              count: localResults.length,
            };
          } else if (
            localResults &&
            localResults.status === "success" &&
            Array.isArray(localResults.data)
          ) {
            return localResults;
          }
        } catch (error) {
          console.log("本地搜索失败，使用vvapi", error);
        }
      }

      const vvapiUrl = `${CONFIG.apiBaseUrl}/search?query=${encodeURIComponent(query)}&min_ratio=${minRatio}&min_similarity=${minSimilarity}`;

      try {
        console.log("使用普通搜索:", vvapiUrl);
        const response = await fetch(vvapiUrl);
        if (!response.ok)
          throw new Error(
            `API 请求失败: ${response.status} ${response.statusText}`,
          );

        const text = await response.text();
        const lines = text.trim().split("\n");
        const results = [];

        for (const line of lines) {
          try {
            if (line.trim()) {
              const item = JSON.parse(line);
              results.push(item);
            }
          } catch (e) {}
        }

        return {
          status: "success",
          data: results,
          count: results.length,
        };
      } catch (error) {
        throw error;
      }
    }

    const emuUrl = `${CONFIG.semanticApiUrl}/search?query=${encodeURIComponent(query)}&min_ratio=${minRatio}&min_similarity=${minSimilarity}&rag=true`;

    try {
      console.log("使用语义搜索:", emuUrl);
      const response = await fetch(emuUrl);
      if (!response.ok)
        throw new Error(
          `API 请求失败: ${response.status} ${response.statusText}`,
        );

      const text = await response.text();
      const lines = text.trim().split("\n");
      const results = [];

      for (const line of lines) {
        try {
          if (line.trim()) {
            const item = JSON.parse(line);
            
            if (item.filename && !item.filename.endsWith('.json')) {
              item.filename = item.filename + '.json';
            }
            
            results.push(item);
          }
        } catch (e) {}
      }

      return {
        status: "success",
        data: results,
        count: results.length,
      };
    } catch (error) {
      throw error;
    }
  }
}
async function handleSearch(event) {
  event.preventDefault();
  const query = document.getElementById("query").value.trim();
  if (!query) return;
  const minRatio = parseInt(document.getElementById("minRatio").value) || 50;
  const minSimilarity =
    parseFloat(document.getElementById("minSimilarity").value) || 0;
  const searchForm = document.getElementById("searchForm");
  searchForm.classList.add("searching");
  startNaturalLoadingBar();

  try {
    const results = await SearchController.performSearch(
      query,
      minRatio,
      minSimilarity,
    );

    if (results && results.status === "success") {
      AppState.cachedResults = results.data;
      AppState.hasMoreResults = results.data.length > AppState.itemsPerPage;
      AppState.displayedCount = 0;

      displayResults(results);
      completeLoadingBar();
    } else {
      throw new Error("Invalid search results format");
    }
  } catch (error) {
    console.error("Search error:", error);
    document.getElementById("errorDisplay").textContent =
      `搜索失败: ${error.message}`;
    document.getElementById("errorDisplay").style.display = "block";
    completeLoadingBar();
  } finally {
    enableKeywordTags();
    searchForm.classList.remove("searching");
  }
}
async function initializeApp() {
  try {
    await dbStorage.init().catch((error) => {});
    mapping = await loadMapping();
    initializeScrollListener();

    for (let i = 0; i <= CONFIG.indexPreloadCount; i++) {
      indexCache.preload(i, CONFIG.imageBaseUrl).catch((error) => {});
    }

    if (
      window.subtitleDB &&
      !window.subtitleDB.isLoaded &&
      !window.subtitleDB.isLoading
    ) {
      window.subtitleDB
        .load()
        .then(() => {
          AppState.dbLoaded = true;
        })
        .catch((error) => {
          setTimeout(() => {
            window.subtitleDB.load().catch((err) => {});
          }, 3000);
        });
    }

    document
      .getElementById("searchForm")
      .addEventListener("submit", async (e) => {
        e.preventDefault();
        if (AppState.isSearching) return;
        AppState.isSearching = true;
        try {
          await handleSearch(e);
        } finally {
          AppState.isSearching = false;
        }
      });

    document.getElementById("query").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (AppState.isSearching) return;
        document
          .getElementById("searchForm")
          .dispatchEvent(new Event("submit"));
      }
    });

    document
      .getElementById("refreshDiv")
      .addEventListener("click", function () {
        location.reload();
      });

    document
      .getElementById("semanticToggle")
      .addEventListener("click", function () {
        this.classList.toggle("active");
      });
  } catch (error) {}
}
document.addEventListener("DOMContentLoaded", () => {
  const loadingBar = document.getElementById("loadingBar");
  if (loadingBar) {
    loadingBar.style.width = "0%";
    loadingBar.style.display = "none";
  }
  initializeApp();
  const toggleButton = document.getElementById("toggleAdvancedOptions");
  const advancedOptions = document.getElementById("advancedOptions");
  toggleButton.addEventListener("click", () => {
    const isExpanded = advancedOptions.classList.contains("show");
    if (!isExpanded) {
      advancedOptions.style.transition = "none";
      advancedOptions.classList.add("show");
      const height = advancedOptions.scrollHeight;
      advancedOptions.classList.remove("show");
      void advancedOptions.offsetHeight;
      advancedOptions.style.transition = "";
      advancedOptions.style.maxHeight = height + "px";
      advancedOptions.classList.add("show");
    } else {
      advancedOptions.style.maxHeight = "0";
      advancedOptions.classList.remove("show");
    }
    toggleButton.classList.toggle("active");
    toggleButton.setAttribute("aria-expanded", !isExpanded);
  });
  
  const semanticToggle = document.getElementById("semanticToggle");
  const semanticTooltip = document.getElementById("semanticTooltip");
  
  semanticToggle.addEventListener("mouseenter", () => {
    semanticTooltip.classList.add("visible");
  });
  
  semanticToggle.addEventListener("mouseleave", () => {
    semanticTooltip.classList.remove("visible");
  });
  
  const watermarkToggle = document.getElementById("watermarkToggle");
  watermarkToggle.addEventListener("change", () => {
    AppState.showWatermark = watermarkToggle.checked;
    if (window.canvasRenderQueue)
      window.canvasRenderQueue.forEach((canvas) => {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(canvas.originalCanvas, 0, 0);
        if (AppState.showWatermark && watermarkLoaded) {
          const watermarkScale = (canvas.width * 0.25) / watermarkImage.width;
          const watermarkWidth = watermarkImage.width * watermarkScale;
          const watermarkHeight = watermarkImage.height * watermarkScale;
          ctx.drawImage(
            watermarkImage,
            canvas.width - watermarkWidth - 5,
            canvas.height - watermarkHeight - 5,
            watermarkWidth,
            watermarkHeight,
          );
        }
      });
  });
});
function displayResults(data, append = false) {
  const resultsDiv = document.getElementById("results");
  const keywordsContainer = document.getElementById("keywordsContainer");

  document.getElementById("errorDisplay").style.display = "none";

  if (!append) {
    resultsDiv.innerHTML = "";
    AppState.displayedCount = 0;
    keywordsContainer.innerHTML = "";
    keywordsContainer.classList.remove("show");
  }

  if (!append && data.data.length > 0 && data.data[0].type === "keywords") {
    const keywords = data.data[0].keywords;
    if (keywords && keywords.length > 0) {
      keywordsContainer.innerHTML = `
        <div class="keywords-tags">
          ${keywords.map(keyword => `<span class="keyword-tag">${keyword}</span>`).join("")}
        </div>
      `;
      keywordsContainer.classList.add("show");
      
      keywordsContainer.querySelectorAll('.keyword-tag').forEach(tag => {
        tag.addEventListener('click', () => {
          if (AppState.isSearching) return;
          
          const keyword = tag.textContent;
          document.getElementById('query').value = keyword;
          
          keywordsContainer.querySelectorAll('.keyword-tag').forEach(t => {
            t.classList.add('disabled');
          });
          
          document.getElementById('searchForm').dispatchEvent(new Event('submit'));
        });
      });
    }
    
    data.data = data.data.slice(1);
  }

  if (data.data && data.data.length === 1 && data.data[0].count === 0) {
    const noResultData = data.data[0];
    console.log("No results case:", {
      message: noResultData.message,
      suggestions: noResultData.suggestions,
    });

    if (!append) {
      const message =
        noResultData.message ||
        `未找到与 "${document.getElementById("query").value.trim()}" 匹配的结果`;
      const suggestions = noResultData.suggestions || [
        "检查输入是否正确",
        `尝试降低最小匹配率（当前：${document.getElementById("minRatio").value}%）`,
        `尝试降低最小相似度（当前：${document.getElementById("minSimilarity").value}）`,
        "尝试使用更简短的关键词",
      ];

      resultsDiv.innerHTML = `
                <div class="error-message">
                    <h3>${message}</h3>
                    <p>建议：</p>
                    <ul>
                        ${suggestions.map((suggestion) => `<li>${suggestion}</li>`).join("")}
                    </ul>
                </div>`;
    }
    AppState.hasMoreResults = false;
    return;
  }

  const fragment = document.createDocumentFragment();
  const startIndex = AppState.displayedCount;
  const endIndex = Math.min(
    startIndex + AppState.itemsPerPage,
    data.data.length,
  );
  const newResults = data.data.slice(startIndex, endIndex);

  AppState.hasMoreResults = endIndex < data.data.length;

  const cards = newResults
    .map((result) => {
      if (!result || typeof result !== "object") return null;
      const card = document.createElement("div");
      card.className = "result-card";
      card.addEventListener("click", () => handleCardClick(result));
      card.style.cursor = "pointer";

      const episodeMatch = result.filename
        ? result.filename.match(/\[P(\d+)\]/)
        : null;
      const timeMatch = result.timestamp
        ? result.timestamp.match(/^(\d+)m(\d+)s$/)
        : null;
      const cleanFilename = result.filename
        ? result.filename
            .replace(/\[P(\d+)\].*?\s+/, "P$1 ")
            .replace(/\.json$/, "")
            .trim()
        : "";

      const cardContent = `
            <div class="result-content">
                <h3>${episodeMatch ? `<span class="tag">${episodeMatch[1]}</span>${cleanFilename.replace(/P\d+/, "").trim()}` : cleanFilename}</h3>
                <p class="result-text">${result.text || ""}</p>
                ${
                  result.timestamp
                    ? `
                <p class="result-meta">
                    ${result.timestamp} \u00b7
                    \u5339\u914d\u5ea6 ${result.match_ratio ? parseFloat(result.match_ratio).toFixed(1) : 0}% \u00b7
                    \u76f8\u4f3c\u5ea6 ${result.similarity ? (result.similarity * 100).toFixed(1) : 0}%
                </p>`
                    : ""
                }
            </div>
        `;

      card.innerHTML = cardContent;
      return card;
    })
    .filter(Boolean);

  cards.forEach((card) => fragment.appendChild(card));
  resultsDiv.appendChild(fragment);

  requestAnimationFrame(() => {
    cards.forEach((card, index) => {
      const result = newResults[index];
      loadPreviewImage(card, result);
    });
  });

  AppState.displayedCount = endIndex;

  if (AppState.hasMoreResults) {
    let trigger = document.getElementById("scroll-trigger");
    if (!trigger) {
      trigger = document.createElement("div");
      trigger.id = "scroll-trigger";
      trigger.style.cssText = "height: 20px; margin: 20px 0;";

      if (window.currentObserver) {
        window.currentObserver.observe(trigger);
      }
    }
    resultsDiv.appendChild(trigger);
  }
}
async function loadPreviewImage(card, result) {
  const episodeMatch = result.filename?.match(/\[P(\d+)\]/);
  const timeMatch = result.timestamp?.match(/^(\d+)m(\d+)s$/);

  if (!episodeMatch || !timeMatch) return;

  const episodeNum = parseInt(episodeMatch[1], 10);
  const minutes = parseInt(timeMatch[1]);
  const seconds = parseInt(timeMatch[2]);
  const totalSeconds = minutes * 60 + seconds;

  const imgContainer = document.createElement("div");
  imgContainer.className = "preview-frame-container";

  const placeholder = document.createElement("div");
  placeholder.className = "preview-frame-placeholder";
  imgContainer.appendChild(placeholder);
  card.insertBefore(imgContainer, card.firstChild);

  try {
    const imageBlob = await extractFrame(
      episodeNum,
      totalSeconds,
      CONFIG.imageBaseUrl,
    );
    const imageUrl = URL.createObjectURL(imageBlob);
    const img = new Image();
    img.src = imageUrl;
    img.className = "preview-frame";
    img.decoding = "async";

    img.onerror = () => {
      console.error("Failed to load preview image");
      imgContainer.remove();
      URL.revokeObjectURL(imageUrl);
    };

    img.onload = () => {
      const originalCanvas = document.createElement("canvas");
      originalCanvas.width = img.width;
      originalCanvas.height = img.height;

      try {
        const originalCtx = originalCanvas.getContext("2d");
        if (!originalCtx) {
          throw new Error("Failed to get canvas context");
        }

        originalCtx.drawImage(img, 0, 0);

        const displayCanvas = document.createElement("canvas");
        displayCanvas.width = img.width;
        displayCanvas.height = img.height;
        displayCanvas.className = "preview-frame";
        displayCanvas.originalCanvas = originalCanvas;

        const renderCanvas = () => {
          const ctx = displayCanvas.getContext("2d");
          if (!ctx) {
            throw new Error("Failed to get display canvas context");
          }

          ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
          ctx.drawImage(originalCanvas, 0, 0);

          if (watermarkLoaded && AppState.showWatermark) {
            const watermarkScale =
              (displayCanvas.width * 0.25) / watermarkImage.width;
            const watermarkWidth = watermarkImage.width * watermarkScale;
            const watermarkHeight = watermarkImage.height * watermarkScale;
            ctx.drawImage(
              watermarkImage,
              displayCanvas.width - watermarkWidth - 5,
              displayCanvas.height - watermarkHeight - 5,
              watermarkWidth,
              watermarkHeight,
            );
          }
        };

        renderCanvas();

        if (!window.canvasRenderQueue) {
          window.canvasRenderQueue = new Set();
        }
        window.canvasRenderQueue.add(displayCanvas);

        displayCanvas.addEventListener("click", (e) => {
          e.stopPropagation();
          displayCanvas.toBlob((blob) => {
            if (!blob) {
              console.error("Failed to create image blob");
              return;
            }
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `VV_${result.filename.replace(/[^\w\s-]/g, "")}_${result.timestamp}.png`;
            a.click();
            URL.revokeObjectURL(url);
          }, "image/png");
        });
        
        // 添加复制按钮
        const copyButton = document.createElement("button");
        copyButton.className = "copy-button";
        copyButton.title = "复制图片";
        copyButton.innerHTML = `<img src="copy.svg" alt="复制" class="copy-icon">`;
        
        copyButton.addEventListener("click", (e) => {
          e.stopPropagation();
          displayCanvas.toBlob(async (blob) => {
            if (!blob) {
              console.error("Failed to create image blob");
              return;
            }
            try {
              // 尝试使用Clipboard API复制图片
              await navigator.clipboard.write([
                new ClipboardItem({
                  [blob.type]: blob
                })
              ]);
              
              // 显示成功提示
              const toast = document.createElement("div");
              toast.className = "copy-toast";
              toast.textContent = "已复制到剪贴板";
              document.body.appendChild(toast);
              
              // 2秒后移除提示
              setTimeout(() => {
                toast.classList.add("fade-out");
                setTimeout(() => toast.remove(), 300);
              }, 2000);
              
            } catch (err) {
              console.error("复制失败:", err);
              alert("复制失败，请使用更新的浏览器或手动保存图片");
            }
          }, "image/png");
        });
        
        imgContainer.appendChild(copyButton);
        imgContainer.appendChild(displayCanvas);
        setTimeout(() => {
          displayCanvas.classList.add("loaded");
          placeholder.style.opacity = "0";
          setTimeout(() => placeholder.remove(), 300);
        }, 50);
      } catch (error) {
        console.error("Canvas error:", error);
        imgContainer.remove();
      }

      URL.revokeObjectURL(imageUrl);
    };
  } catch (error) {
    console.error("加载预览图失败:", error);
    imgContainer.remove();
  }
}
function getEpisodeUrl(filename) {
  for (let key in mapping) if (mapping[key] === filename) return key;
  return null;
}
function startNaturalLoadingBar() {
  const loadingBar = document.getElementById("loadingBar");
  loadingBar.style.transition = "";
  loadingBar.style.width = "0%";
  loadingBar.style.display = "block";
  if (loadingBar.interval) clearInterval(loadingBar.interval);
  let progress = 0;
  const targetProgress = 95;
  let speed = 0.5;
  loadingBar.interval = setInterval(() => {
    if (progress < 30) speed = 0.8;
    else if (progress < 60) speed = 0.4;
    else if (progress < 80) speed = 0.2;
    else speed = 0.1;
    progress += speed;
    if (progress >= targetProgress) {
      clearInterval(loadingBar.interval);
      progress = targetProgress;
    }
    loadingBar.style.width = `${progress}%`;
  }, 50);
}
function completeLoadingBar() {
  const loadingBar = document.getElementById("loadingBar");
  clearInterval(loadingBar.interval);
  loadingBar.style.transition = "width 0.3s ease-out";
  loadingBar.style.width = "100%";
}
function initializeScrollListener() {
  if (window.currentObserver) window.currentObserver.disconnect();

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (
          entry.isIntersecting &&
          AppState.hasMoreResults &&
          !AppState.isSearching
        ) {
          if (AppState.cachedResults.length > AppState.displayedCount) {
            displayResults(
              {
                status: "success",
                data: AppState.cachedResults,
                count: AppState.cachedResults.length,
              },
              true,
            );
          }
        }
      });
    },
    { root: null, rootMargin: "200px", threshold: 0.1 },
  );

  window.currentObserver = observer;

  const oldTrigger = document.getElementById("scroll-trigger");
  if (oldTrigger) oldTrigger.remove();

  const trigger = document.createElement("div");
  trigger.id = "scroll-trigger";
  trigger.style.cssText = "height: 20px; margin: 20px 0;";
  document.getElementById("results").appendChild(trigger);

  observer.observe(trigger);
}
function handleCardClick(result) {
  const episodeMatch = result.filename.match(/\[P(\d+)\]/);
  const timeMatch = result.timestamp.match(/^(\d+)m(\d+)s$/);
  if (episodeMatch && timeMatch) {
    const episodeNum = parseInt(episodeMatch[1], 10);
    const minutes = parseInt(timeMatch[1]);
    const seconds = parseInt(timeMatch[2]);
    const totalSeconds = minutes * 60 + seconds;
    for (const [url, filename] of Object.entries(mapping))
      if (filename === result.filename) {
        const videoUrl = `https://www.bilibili.com${url}?t=${totalSeconds}`;
        window.open(videoUrl, "_blank");
        break;
      }
  }
}
function enableKeywordTags() {
  const keywordsContainer = document.getElementById("keywordsContainer");
  keywordsContainer.querySelectorAll('.keyword-tag').forEach(tag => {
    tag.classList.remove('disabled');
  });
}