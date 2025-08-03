'use strict'; class SubtitleDatabase {
    constructor() { this.db = null; this.isLoading = false; this.isLoaded = false; this.loadPromise = null; this.progressCallback = null; this.version = "1.0.1" } async clearCache() { try { await dbStorage.removeItem("databases", "subtitleDB"); await dbStorage.setItem("databases", "version", this.version); this.db = null; this.isLoaded = false; this.isLoading = false; this.loadPromise = null; return true } catch (error) { throw error; } } getDebugInfo() {
        return {
            version: this.version, isLoaded: this.isLoaded, isLoading: this.isLoading,
            recordCount: this.db ? this.db.length : 0, memoryUsage: this.db ? JSON.stringify(this.db).length : 0, cacheStatus: this.isLoaded ? "\u5df2\u52a0\u8f7d" : this.isLoading ? "\u52a0\u8f7d\u4e2d" : "\u672a\u52a0\u8f7d"
        }
    } printDebugInfo() { const info = this.getDebugInfo() } async checkVersion() {
        try {
            const storedVersion = await dbStorage.getItem("databases", "version"); if (!storedVersion) { await dbStorage.setItem("databases", "version", this.version); return false } if (storedVersion !== this.version) {
                await this.clearCache(); await dbStorage.setItem("databases",
                    "version", this.version); return true
            } return false
        } catch (error) { return false }
    } async load(progressCallback = null) {
        if (this.isLoaded && this.db && this.db.length > 0) return true;
        if (this.isLoading) return this.loadPromise;
        this.isLoading = true;
        this.progressCallback = progressCallback;
        await this.checkVersion();
        
        try {
            const cachedDB = await dbStorage.getItem("databases", "subtitleDB");
            if (cachedDB) {
                if (Array.isArray(cachedDB) && cachedDB.length > 0) {
                    this.db = cachedDB;
                    this.isLoaded = true;
                    this.isLoading = false;
                    return true;
                }
            }
        } catch (e) { }
        
        this.loadPromise = new Promise(async (resolve, reject) => {
            try {
                const checkResponse = await fetch("https://vvdb.cicada000.work/subtitle_db", {
                    method: "HEAD",
                    referrerPolicy: 'no-referrer',
                    mode: 'cors',
                    credentials: 'omit'
                });
                
                if (!checkResponse.ok) throw new Error(`Database file not found: ${checkResponse.status}`);
                
                const response = await fetch("https://vvdb.cicada000.work/subtitle_db", {
                    referrerPolicy: 'no-referrer',
                    mode: 'cors',
                    credentials: 'omit'
                });
                
                if (!response.ok) throw new Error(`Failed to load database: ${response.status}`);
                
                const contentLength = response.headers.get("Content-Length");
                const reader = response.body.getReader();
                let receivedLength = 0;
                const chunks = [];
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    receivedLength += value.length;
                }
                
                const chunksAll = new Uint8Array(receivedLength);
                let position = 0;
                for (const chunk of chunks) {
                    chunksAll.set(chunk, position);
                    position += chunk.length
                }
                const ds =
                    new DecompressionStream("gzip");
                const decompressedStream = (new Response(chunksAll)).body.pipeThrough(ds);
                const decompressedData = await (new Response(decompressedStream)).text();
                const parsedData = JSON.parse(decompressedData);
                if (!parsedData || !Array.isArray(parsedData) || parsedData.length === 0) throw new Error("Invalid or empty database format");
                this.db = parsedData;
                this.isLoaded = true;
                this.isLoading = false;
                try { await dbStorage.setItem("databases", "subtitleDB", this.db) } catch (e) { }
                if (window.subtitleDB !==
                    this) window.subtitleDB = this;
                resolve(true)
            } catch (error) {
                this.isLoading = false;
                this.isLoaded = false;
                this.db = null;
                reject(error)
            }
        });
        return this.loadPromise
    } lcsRatio(str1, str2) { str1 = str1.toLowerCase(); str2 = str2.toLowerCase(); if (!str1 || !str2) return 0; const m = str1.length; const n = str2.length; const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0)); for (let i = 1; i <= m; i++)for (let j = 1; j <= n; j++)if (str1[i - 1] === str2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1; else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]); return dp[m][n] / m * 100 } multiWordLcsRatio(queryWords,
        text) {
            text = text.toLowerCase(); const totalQueryLength = queryWords.reduce((sum, word) => sum + word.length, 0); const usedChars = (new Array(text.length)).fill(false); let totalMatched = 0; for (const word of queryWords) {
                const wordLower = word.toLowerCase(); let foundMatch = false; let startPos = 0; while (true) {
                    const pos = text.indexOf(wordLower, startPos); if (pos === -1) break; const endPos = pos + wordLower.length; let canUse = true; for (let i = pos; i < endPos; i++)if (usedChars[i]) { canUse = false; break } if (canUse) {
                        for (let i = pos; i < endPos; i++)usedChars[i] =
                            true; totalMatched += wordLower.length; foundMatch = true; break
                    } startPos = pos + 1
                }
            } return totalMatched / totalQueryLength * 100
    } search(query, minRatio = 50, minSimilarity = 0) {
        if (!this.isLoaded || !this.db || this.db.length === 0) {
            return {
                status: "success",
                data: [],
                count: 0,
                message: "数据库未加载或为空",
                suggestions: [
                    "请等待数据库加载完成",
                    "如果问题持续，请刷新页面"
                ]
            };
        } 
        const startTime = performance.now(); 
        query = query.toLowerCase(); 
        const hasSpaces = query.includes(" ") || query.includes("%20"); 
        const queryWords = hasSpaces ? query.replace(/%20/g, " ").split(/\s+/).filter(Boolean) : [query]; 
        const filteredBySimiliarity = this.db.filter(item => item.s >= minSimilarity); 
        
        if (window.Worker && queryWords.length > 1) return new Promise(resolve => {
            const worker = new Worker("search_worker.js"); 
            worker.onmessage = e => {
                const workerResults = e.data; 
                worker.terminate(); 
                
                if (workerResults.status === "success") {
                    if (workerResults.data.length === 0) {
                        resolve({
                            status: "success",
                            data: [],
                            count: 0,
                            message: `未找到与 '${query}' 匹配的结果`,
                            suggestions: [
                                "检查输入是否正确",
                                `尝试降低最小匹配率（当前：${minRatio}%）`,
                                `尝试降低最小相似度（当前：${minSimilarity}）`,
                                "尝试使用更简短的关键词"
                            ]
                        });
                    } else {
                        resolve(workerResults);
                    }
                } else {
                    resolve({
                        status: "error",
                        data: [],
                        count: 0,
                        message: workerResults.message || "搜索失败",
                        suggestions: ["请重试"]
                    });
                }
            }; 
            
            worker.onerror = (error) => {
                console.error("Worker error:", error);
                resolve({ 
                    status: "error", 
                    message: "Search failed", 
                    data: [], 
                    count: 0 
                });
            };
            
            worker.postMessage({ db: filteredBySimiliarity, queryWords, minRatio });
        }); 
        
        const results = filteredBySimiliarity.map(item => { 
            const matchRatio = hasSpaces ? this.multiWordLcsRatio(queryWords, item.x) : this.lcsRatio(query, item.x); 
            return { 
                matchRatio, 
                item, 
                exactMatch: queryWords.every(word => item.x.toLowerCase().includes(word)) 
            }; 
        }).filter(result => result.matchRatio >= minRatio); 
        
        results.sort((a, b) => {
            if (b.matchRatio !== a.matchRatio) return b.matchRatio - a.matchRatio; 
            return b.exactMatch - a.exactMatch;
        }); 
        
        const apiResults = results.map(result => ({ 
            filename: result.item.f, 
            timestamp: result.item.t, 
            similarity: result.item.s, 
            text: result.item.x, 
            match_ratio: result.matchRatio, 
            exact_match: result.exactMatch 
        })); 
        
        if (apiResults.length === 0) {
            return {
                status: "success",
                data: [{  // 包装在数组中
                    status: "success",
                    data: [],
                    count: 0,
                    folder: "subtitle",
                    max_results: "unlimited",
                    message: `未找到与 '${query}' 匹配的结果`,
                    suggestions: [
                        "检查输入是否正确",
                        `尝试降低最小匹配率（当前：${minRatio}%）`,
                        `尝试降低最小相似度（当前：${minSimilarity}）`,
                        "尝试使用更简短的关键词"
                    ]
                }],
                count: 1  // 数组长度为1
            };
        }

        return {
            status: "success",
            data: apiResults,
            count: apiResults.length
        };
    }
}
window.dbDebug = { clearCache: async () => { try { await window.subtitleDB.clearCache() } catch (error) { } }, info: () => { window.subtitleDB.printDebugInfo() }, reload: async () => { try { await window.subtitleDB.load() } catch (error) { } }, help: () => { } }; window.subtitleDB = new SubtitleDatabase; fetch("https://vvdb.cicada000.work/subtitle_db", { 
    method: "HEAD",
    referrerPolicy: 'no-referrer',
    mode: 'cors',
    credentials: 'omit'
}).then(response => {
    if (response.ok) window.subtitleDB.load().catch(error => { })
}).catch(error => { });