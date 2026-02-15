// accuracyTracker.js
// Theo dõi độ chính xác của các mô hình và ensemble

const simpleStats = require('simple-statistics');

class AccuracyTracker {
    constructor() {
        // Lưu các dự đoán đã thực hiện, chờ kết quả thực tế
        // Mỗi bàn có một mảng các prediction objects
        this.predictions = {
            tx: [],
            md5: []
        };

        // Lưu độ chính xác gần đây cho từng mô hình (để tính trọng số động)
        // Mỗi mô hình có mảng accuracy 20 lần gần nhất
        this.modelRecentAccuracy = {
            // Các mô hình sẽ được đăng ký sau
        };
    }

    // Đăng ký một mô hình (gọi từ predictor.js khi khởi tạo)
    registerModel(modelName) {
        if (!this.modelRecentAccuracy[modelName]) {
            this.modelRecentAccuracy[modelName] = [];
        }
    }

    // Ghi nhận một dự đoán vừa thực hiện (chưa có kết quả thực)
    addPrediction(tableName, prediction) {
        if (!this.predictions[tableName]) {
            this.predictions[tableName] = [];
        }
        // Gán thêm trường để sau này cập nhật kết quả
        const predRecord = {
            ...prediction,
            actualResult: null,
            correct: null,
            timestamp: new Date().toISOString()
        };
        this.predictions[tableName].push(predRecord);

        // Giới hạn lưu 500 dự đoán gần nhất để tránh tràn
        if (this.predictions[tableName].length > 500) {
            this.predictions[tableName] = this.predictions[tableName].slice(-500);
        }

        return predRecord;
    }

    // Khi có kết quả thực tế của một phiên, cập nhật vào dự đoán tương ứng
    recordActual(tableName, sid, actualResult) {
        const predictions = this.predictions[tableName];
        // Tìm dự đoán có phien_du_doan === sid
        const pred = predictions.find(p => p.phien_du_doan === sid);
        if (!pred) return false;

        pred.actualResult = actualResult;
        pred.correct = (pred.ket_qua_du_doan === actualResult) ? 1 : 0;

        // Cập nhật độ chính xác gần đây cho từng mô hình
        // (Phần này sẽ được gọi riêng từ predictor khi có kết quả)
        return true;
    }

    // Tính accuracy tổng thể cho một bàn
    getOverallAccuracy(tableName) {
        const preds = this.predictions[tableName] || [];
        const validPreds = preds.filter(p => p.correct !== null);
        if (validPreds.length === 0) return 0;
        const correctCount = validPreds.reduce((sum, p) => sum + p.correct, 0);
        return (correctCount / validPreds.length) * 100;
    }

    // Tính accuracy 50 phiên gần nhất
    getRecentAccuracy(tableName, limit = 50) {
        const preds = this.predictions[tableName] || [];
        const validPreds = preds.filter(p => p.correct !== null).slice(-limit);
        if (validPreds.length === 0) return 0;
        const correctCount = validPreds.reduce((sum, p) => sum + p.correct, 0);
        return (correctCount / validPreds.length) * 100;
    }

    // Tính precision, recall, F1, Brier score
    getDetailedStats(tableName) {
        const preds = this.predictions[tableName] || [];
        const valid = preds.filter(p => p.correct !== null);
        if (valid.length === 0) {
            return {
                precision: 0,
                recall: 0,
                f1: 0,
                brier: 0,
                totalPredictions: 0
            };
        }

        let tp = 0, fp = 0, tn = 0, fn = 0;
        let brierSum = 0;

        valid.forEach(p => {
            const actual = p.actualResult; // 'tài' hoặc 'xỉu'
            const predicted = p.ket_qua_du_doan;
            const prob = p.ti_le_tai / 100; // xác suất tài dự đoán

            // Brier score: (prob - actual)^2 với actual là 1 nếu tài, 0 nếu xỉu
            const actualBinary = (actual === 'tài') ? 1 : 0;
            brierSum += Math.pow(prob - actualBinary, 2);

            if (predicted === 'tài' && actual === 'tài') tp++;
            else if (predicted === 'tài' && actual === 'xỉu') fp++;
            else if (predicted === 'xỉu' && actual === 'xỉu') tn++;
            else if (predicted === 'xỉu' && actual === 'tài') fn++;
        });

        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = (2 * precision * recall) / (precision + recall) || 0;
        const brier = brierSum / valid.length;

        return {
            precision: precision * 100,
            recall: recall * 100,
            f1: f1 * 100,
            brier: brier,
            totalPredictions: valid.length
        };
    }

    // Lấy độ chính xác gần đây cho từng mô hình (để tính trọng số)
    getModelRecentAccuracy(modelName, limit = 20) {
        const accs = this.modelRecentAccuracy[modelName] || [];
        if (accs.length === 0) return 0.5; // default
        const recent = accs.slice(-limit);
        return recent.reduce((a, b) => a + b, 0) / recent.length;
    }

    // Cập nhật accuracy cho mô hình khi có kết quả thực tế (gọi từ predictor)
    updateModelAccuracy(modelName, correct) {
        if (!this.modelRecentAccuracy[modelName]) {
            this.modelRecentAccuracy[modelName] = [];
        }
        this.modelRecentAccuracy[modelName].push(correct ? 1 : 0);
        // Giữ tối đa 100
        if (this.modelRecentAccuracy[modelName].length > 100) {
            this.modelRecentAccuracy[modelName] = this.modelRecentAccuracy[modelName].slice(-100);
        }
    }
}

module.exports = new AccuracyTracker();
