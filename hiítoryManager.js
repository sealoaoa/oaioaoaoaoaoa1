// historyManager.js
// Quản lý lịch sử phiên cho từng bàn (tx, md5)

class HistoryManager {
    constructor(maxSize = 1000) {
        this.storage = {
            tx: [],    // mảng các phiên bàn TX
            md5: []    // mảng các phiên bàn MD5
        };
        this.maxSize = maxSize; // giữ tối đa 1000 phiên gần nhất
    }

    // Thêm một phiên mới vào lịch sử của bàn, kiểm tra trùng sid
    addSession(tableName, session) {
        if (!this.storage[tableName]) {
            this.storage[tableName] = [];
        }

        // Kiểm tra trùng lặp dựa trên sid
        const exists = this.storage[tableName].some(s => s.sid === session.sid);
        if (exists) return false;

        // Thêm timestamp nếu chưa có
        if (!session.thoi_gian) {
            session.thoi_gian = new Date().toISOString();
        }

        // Thêm vào mảng
        this.storage[tableName].push(session);

        // Sắp xếp theo sid tăng dần (đảm bảo thứ tự thời gian)
        this.storage[tableName].sort((a, b) => a.sid - b.sid);

        // Giới hạn kích thước
        if (this.storage[tableName].length > this.maxSize) {
            this.storage[tableName] = this.storage[tableName].slice(-this.maxSize);
        }

        return true;
    }

    // Lấy lịch sử của bàn, có thể giới hạn số lượng
    getHistory(tableName, limit = null) {
        if (!this.storage[tableName]) return [];
        if (limit && limit > 0) {
            return this.storage[tableName].slice(-limit);
        }
        return this.storage[tableName];
    }

    // Lấy phiên gần nhất (sid lớn nhất)
    getLatestSession(tableName) {
        const hist = this.storage[tableName];
        if (!hist || hist.length === 0) return null;
        return hist[hist.length - 1];
    }

    // Lấy sid lớn nhất hiện tại
    getLatestSid(tableName) {
        const latest = this.getLatestSession(tableName);
        return latest ? latest.sid : null;
    }

    // Lấy toàn bộ lịch sử (cho mục đích debug)
    getAllHistory() {
        return this.storage;
    }

    // Xóa dữ liệu cũ (có thể dùng khi cần reset)
    clear(tableName) {
        if (tableName) {
            this.storage[tableName] = [];
        } else {
            this.storage = { tx: [], md5: [] };
        }
    }
}

module.exports = new HistoryManager();
