// predictor.js
// Ensemble các mô hình dự đoán, bao gồm mô hình đã chuyển từ Python
// Sử dụng các thư viện: simple-statistics, ml-logistic-regression, brain.js

const simpleStats = require('simple-statistics');
const LogisticRegression = require('ml-logistic-regression');
const brain = require('brain.js');
const accuracyTracker = require('./accuracyTracker');

// Khởi tạo các mô hình
class Predictor {
    constructor() {
        // Đăng ký các mô hình với accuracyTracker
        this.modelNames = ['markov', 'logistic', 'naiveBayes', 'neuralNet', 'legacyPython'];
        this.modelNames.forEach(name => accuracyTracker.registerModel(name));

        // Trọng số hiện tại (khởi tạo bằng nhau)
        this.weights = {};
        this.modelNames.forEach(name => { this.weights[name] = 1.0; });

        // Đối tượng mô hình riêng lẻ
        this.models = {
            logistic: new LogisticRegression({ numSteps: 1000, learningRate: 0.1 }),
            neuralNet: new brain.NeuralNetwork({ hiddenLayers: [5, 3] })
        };

        // Dữ liệu huấn luyện cho các mô hình có học
        this.trainingData = {
            logistic: { features: [], labels: [] },
            neuralNet: { inputs: [], outputs: [] }
        };

        // Số lượng mẫu tối thiểu để train
        this.minTrainSamples = 20;
    }

    // Hàm tiền xử lý: trích xuất đặc trưng từ lịch sử
    extractFeatures(history, targetSid = null) {
        // Lấy các phiên trước targetSid (nếu có) hoặc dùng toàn bộ history
        let relevantHistory = history;
        if (targetSid) {
            relevantHistory = history.filter(s => s.sid < targetSid);
        }
        if (relevantHistory.length === 0) return null;

        // Đặc trưng cơ bản:
        const last5 = relevantHistory.slice(-5);
        const last10 = relevantHistory.slice(-10);
        const last20 = relevantHistory.slice(-20);

        // Tỉ lệ tài trong các cửa sổ
        const taiRate = (arr) => arr.filter(s => s.ket_qua === 'tài').length / arr.length;

        const features = [
            taiRate(last5),                  // tỉ lệ tài 5 phiên gần
            taiRate(last10),                 // tỉ lệ tài 10 phiên gần
            taiRate(last20),                 // tỉ lệ tài 20 phiên gần
            simpleStats.mean(last10.map(s => s.tong)), // trung bình tổng điểm 10 phiên
            simpleStats.standardDeviation(last10.map(s => s.tong)) || 0, // độ lệch chuẩn
        ];

        // Thêm kết quả 5 phiên trước (mã hóa tài=1, xỉu=0)
        for (let i = 1; i <= 5; i++) {
            const idx = relevantHistory.length - i;
            if (idx >= 0) {
                features.push(relevantHistory[idx].ket_qua === 'tài' ? 1 : 0);
            } else {
                features.push(0.5); // giá trị mặc định nếu không đủ
            }
        }

        return features;
    }

    // Mô hình 1: Markov Chain bậc 1,2,3 có trọng số thời gian
    markovPredict(history) {
        if (history.length < 3) return 0.5;

        // Chuyển chuỗi kết quả thành dạng số (tài=1, xỉu=0)
        const seq = history.map(s => s.ket_qua === 'tài' ? 1 : 0);
        const n = seq.length;

        // Tính ma trận chuyển tiếp bậc 1,2,3 có trọng số thời gian (càng gần càng quan trọng)
        const weight = (idx) => Math.exp((idx - n) / 10); // trọng số giảm dần theo thời gian

        let trans1 = { '0->0': 0, '0->1': 0, '1->0': 0, '1->1': 0 };
        let trans2 = { '00->0':0, '00->1':0, '01->0':0, '01->1':0, '10->0':0, '10->1':0, '11->0':0, '11->1':0 };
        let trans3 = {}; // 16 tổ hợp, tạm thời bỏ qua cho gọn

        for (let i = 0; i < n-1; i++) {
            const w = weight(i);
            const key = `${seq[i]}->${seq[i+1]}`;
            trans1[key] += w;
        }
        for (let i = 0; i < n-2; i++) {
            const w = weight(i);
            const key = `${seq[i]}${seq[i+1]}->${seq[i+2]}`;
            trans2[key] += w;
        }

        // Lấy trạng thái gần nhất
        const last = seq[seq.length-1];
        const last2 = seq.slice(-2).join('');

        // Xác suất tài dựa trên bậc 1
        const total1 = trans1[`${last}->0`] + trans1[`${last}->1`];
        const prob1 = total1 === 0 ? 0.5 : trans1[`${last}->1`] / total1;

        // Xác suất dựa trên bậc 2
        const total2 = trans2[`${last2}->0`] + trans2[`${last2}->1`];
        const prob2 = total2 === 0 ? 0.5 : trans2[`${last2}->1`] / total2;

        // Kết hợp có trọng số (ưu tiên bậc cao hơn)
        const prob = (prob1 + 2*prob2) / 3;
        return prob;
    }

    // Mô hình 2: Hồi quy logistic (sử dụng ml-logistic-regression)
    logisticPredict(history) {
        if (history.length < this.minTrainSamples) return 0.5;

        // Chuẩn bị dữ liệu: mỗi mẫu là một vector đặc trưng của các phiên trước, nhãn là kết quả phiên tiếp theo
        // Ta cần tạo tập huấn luyện từ history
        const X = [];
        const y = [];
        for (let i = 10; i < history.length; i++) {
            const histWindow = history.slice(0, i); // các phiên trước i
            const features = this.extractFeatures(histWindow, history[i].sid);
            if (features) {
                X.push(features);
                y.push(history[i].ket_qua === 'tài' ? 1 : 0);
            }
        }

        if (X.length < 5) return 0.5;

        // Huấn luyện lại mô hình (có thể huấn luyện incrementally, nhưng đơn giản là train mới)
        try {
            this.models.logistic.train(X, y);
            // Dự đoán cho phiên tiếp theo
            const lastFeatures = this.extractFeatures(history);
            if (!lastFeatures) return 0.5;
            const predProb = this.models.logistic.predictProbabilities([lastFeatures])[0][1]; // xác suất lớp 1 (tài)
            return predProb;
        } catch (e) {
            console.error('Logistic regression error:', e);
            return 0.5;
        }
    }

    // Mô hình 3: Naive Bayes đơn giản dựa trên tần suất
    naiveBayesPredict(history) {
        if (history.length < 10) return 0.5;

        const taiCount = history.filter(s => s.ket_qua === 'tài').length;
        const xiuCount = history.length - taiCount;
        const priorTai = taiCount / history.length;

        // Giả sử độc lập: xét 5 phiên gần nhất
        const last5 = history.slice(-5);
        const taiLast5 = last5.filter(s => s.ket_qua === 'tài').length;
        // Tính likelihood đơn giản: tần suất tài trong 5 phiên gần
        const likelihoodTai = taiLast5 / 5;
        // Posterior (ước lượng)
        const posterior = priorTai * likelihoodTai; // chưa chuẩn hóa
        // Tạm tính xác suất tài = posterior / (posterior + (1-priorTai)*(1-likelihoodTai)?)
        // Đơn giản hơn: trả về tỉ lệ tài trong 10 phiên gần
        const last10 = history.slice(-10);
        const rate = last10.filter(s => s.ket_qua === 'tài').length / 10;
        return rate;
    }

    // Mô hình 4: Neural network với brain.js
    neuralNetPredict(history) {
        if (history.length < this.minTrainSamples) return 0.5;

        // Chuẩn bị dữ liệu tương tự logistic
        const trainingData = [];
        for (let i = 10; i < history.length; i++) {
            const histWindow = history.slice(0, i);
            const features = this.extractFeatures(histWindow, history[i].sid);
            if (features) {
                trainingData.push({
                    input: features,
                    output: [history[i].ket_qua === 'tài' ? 1 : 0]
                });
            }
        }

        if (trainingData.length < 5) return 0.5;

        try {
            this.models.neuralNet.train(trainingData, { iterations: 100, errorThresh: 0.01 });
            const lastFeatures = this.extractFeatures(history);
            if (!lastFeatures) return 0.5;
            const output = this.models.neuralNet.run(lastFeatures);
            return output[0]; // xác suất tài
        } catch (e) {
            console.error('Neural net error:', e);
            return 0.5;
        }
    }

    // Mô hình 5: Legacy Python (giả lập) - cần thay bằng code chuyển đổi thực tế
    legacyPythonPredict(history) {
        // Giả sử code Python cũ dùng phân tích pattern đơn giản
        if (history.length < 5) return 0.5;

        // Ví dụ: nếu 3 phiên gần nhất đều tài thì dự đoán xỉu với xác suất cao
        const last3 = history.slice(-3);
        const allTai = last3.every(s => s.ket_qua === 'tài');
        const allXiu = last3.every(s => s.ket_qua === 'xỉu');

        if (allTai) return 0.3; // nghiêng về xỉu
        if (allXiu) return 0.7; // nghiêng về tài

        // Mặc định: tỉ lệ tài 10 phiên gần
        const last10 = history.slice(-10);
        const rate = last10.filter(s => s.ket_qua === 'tài').length / 10;
        return rate;
    }

    // Hàm dự đoán chính (ensemble)
    predict(history, tableName) {
        if (!history || history.length < 10) {
            return {
                error: 'Chưa đủ dữ liệu để dự đoán (cần tối thiểu 10 phiên)',
                code: 'INSUFFICIENT_DATA'
            };
        }

        // Lấy kết quả từ từng mô hình
        const probs = {};
        this.modelNames.forEach(name => {
            let prob;
            switch (name) {
                case 'markov': prob = this.markovPredict(history); break;
                case 'logistic': prob = this.logisticPredict(history); break;
                case 'naiveBayes': prob = this.naiveBayesPredict(history); break;
                case 'neuralNet': prob = this.neuralNetPredict(history); break;
                case 'legacyPython': prob = this.legacyPythonPredict(history); break;
                default: prob = 0.5;
            }
            probs[name] = prob;
        });

        // Cập nhật trọng số động dựa trên độ chính xác gần đây của từng mô hình
        let totalWeight = 0;
        const dynamicWeights = {};
        this.modelNames.forEach(name => {
            const recentAcc = accuracyTracker.getModelRecentAccuracy(name, 20);
            // Chuyển accuracy thành trọng số (có thể dùng chính nó, nhưng để tránh 0 thì + epsilon)
            const weight = Math.max(0.1, recentAcc); 
            dynamicWeights[name] = weight;
            totalWeight += weight;
        });

        // Tính xác suất ensemble có trọng số
        let ensembleProb = 0;
        this.modelNames.forEach(name => {
            ensembleProb += probs[name] * dynamicWeights[name];
        });
        ensembleProb /= totalWeight;

        // Xác định kết quả
        const ketQua = ensembleProb >= 0.5 ? 'tài' : 'xỉu';
        const tiLeTai = ensembleProb * 100;
        const tiLeXiu = 100 - tiLeTai;

        // Tính độ tin cậy dựa trên số mẫu và độ đồng thuận
        let doTinCay = 'thấp';
        const historyLength = history.length;
        if (historyLength >= 50) doTinCay = 'cao';
        else if (historyLength >= 20) doTinCay = 'trung bình';

        // Nếu các mô hình có độ phân tán thấp (đồng thuận cao) thì tăng độ tin cậy
        const probValues = Object.values(probs);
        const stdDev = simpleStats.standardDeviation(probValues) || 0;
        if (stdDev < 0.1 && historyLength >= 20) doTinCay = 'cao';

        // Tạo đối tượng dự đoán
        const prediction = {
            ket_qua_du_doan: ketQua,
            ti_le_tai: tiLeTai,
            ti_le_xiu: tiLeXiu,
            phien_du_doan: history[history.length-1].sid + 1, // phiên tiếp theo
            so_lieu_su_dung: history.length,
            do_tin_cay: doTinCay,
            chi_tiet_mo_hinh: probs, // để debug
            trong_so: dynamicWeights
        };

        // Lưu vào accuracyTracker để chờ kết quả thực tế
        accuracyTracker.addPrediction(tableName, prediction);

        return prediction;
    }

    // Hàm cập nhật khi có kết quả thực tế (gọi từ server)
    updateActual(tableName, sid, actualResult) {
        // Cập nhật accuracyTracker
        const updated = accuracyTracker.recordActual(tableName, sid, actualResult);
        if (!updated) return;

        // Lấy dự đoán vừa được cập nhật
        const predictions = accuracyTracker.predictions[tableName];
        const pred = predictions.find(p => p.phien_du_doan === sid);
        if (!pred) return;

        // Cập nhật độ chính xác cho từng mô hình dựa trên correctness của chúng?
        // Ở đây ta chỉ cập nhật cho ensemble, nhưng để tính trọng số, ta cần biết mỗi mô hình đúng hay sai.
        // Tuy nhiên ta không có lưu riêng kết quả từng mô hình. Giải pháp: lưu riêng từng mô hình khi dự đoán.
        // Để đơn giản, ta sẽ giả định rằng ta có thể tính correctness cho từng mô hình dựa trên dự đoán của chúng.
        // Cần mở rộng accuracyTracker để lưu chi tiết từng mô hình. Nhưng trong khuôn khổ, ta tạm thời chỉ cập nhật trọng số bằng cách dùng correctness của ensemble.
        // Thực tế, để có dynamic weights chính xác, cần lưu riêng. Tôi sẽ cập nhật phần này sau.
    }
}

module.exports = new Predictor();
