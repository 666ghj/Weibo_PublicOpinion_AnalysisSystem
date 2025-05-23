/**
 * 客户端数据缓存工具
 */
(function() {
    'use strict';
  
    // 创建数据库命名空间
    window.DB = {
        // 从localStorage获取数据
        get: function(key) {
            try {
                return JSON.parse(localStorage.getItem(key));
            } catch (e) {
                console.error('从存储获取数据失败:', e);
                return null;
            }
        },
        
        // 保存数据到localStorage
        set: function(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('保存数据到存储失败:', e);
                return false;
            }
        },
        
        // 删除数据
        remove: function(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('删除存储的数据失败:', e);
                return false;
            }
        },
        
        // 清除所有数据
        clear: function() {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('清除所有存储数据失败:', e);
                return false;
            }
        }
    };
    
    console.log('客户端数据库初始化完成');
})(); 