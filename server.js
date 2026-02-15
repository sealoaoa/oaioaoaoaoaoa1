const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const simpleStats = require('simple-statistics');
const LogisticRegression = require('ml-logistic-regression');
const brain = require('brain.js');

// ==================== HISTORY MANAGER ====================
class HistoryManager {
    constructor(maxSize = 1000) {
        this.storage = { tx: [], md5: [] };
        this.maxSize = maxSize;
    }

    addSession(tableName, session) {
        if (!this.storage[tableName]) this.storage[tableName] = [];
        const exists = this.storage[tableName].some(s => s.sid === session.sid);
        if (exists) return false;
        if (!session.thoi_gian) session.thoi_gian = new Date().toISOString();
        this.storage[tableName].push(session);
        this.storage[tableName].sort((a, b) => a.sid - b.sid);
        if (this.storage[tableName].length > this.maxSize)
            this.storage[tableName] = this.storage[tableName].slice(-this.maxSize);
        return true;
    }

    getHistory(tableName, limit = null) {
        if (!this.storage[tableName]) return [];
        if (limit) return this.storage[tableName].slice(-limit);
        return this.storage[tableName];
    }

    getLatestSession(tableName) {
        const hist = this.storage[tableName];
        return hist?.length ? hist[hist.length - 1] : null;
    }

    getLatestSid(tableName) {
        return this.getLatestSession(tableName)?.sid || 0;
    }

    clear(tableName) {
        if (tableName) this.storage[tableName] = [];
        else this.storage = { tx: [], md5: [] };
    }
}

// ==================== ACCURACY TRACKER ====================
class AccuracyTracker {
    constructor() {
        this.predictions = { tx: [], md5: [] };
        this.modelRecentAccuracy = {}; // { modelName: [0/1...] }
    }

    registerModel(modelName) {
        if (!this.modelRecentAccuracy[modelName])
            this.modelRecentAccuracy[modelName] = [];
    }

    addPrediction(tableName, prediction) {
        if (!this.predictions[tableName]) this.predictions[tableName] = [];
        const predRecord = {
            ...prediction,
            actualResult: null,
            correct: null,
            timestamp: new Date().toISOString()
        };
        this.predictions[tableName].push(predRecord);
        if (this.predictions[tableName].length > 500)
            this.predictions[tableName] = this.predictions[tableName].slice(-500);
        return predRecord;
    }

    recordActual(tableName, sid, actualResult) {
        const pred = this.predictions[tableName]?.find(p => p.phien_du_doan === sid);
        if (!pred) return false;
        pred.actualResult = actualResult;
        pred.correct = pred.ket_qua_du_doan === actualResult ? 1 : 0;
        return true;
    }

    getOverallAccuracy(tableName) {
        const valid = this.predictions[tableName]?.filter(p => p.correct !== null) || [];
        if (!valid.length) return 0;
        return (valid.reduce((sum, p) => sum + p.correct, 0) / valid.length) * 100;
    }

    getRecentAccuracy(tableName, limit = 50) {
        const valid = this.predictions[tableName]?.filter(p => p.correct !== null).slice(-limit) || [];
        if (!valid.length) return 0;
        return (valid.reduce((sum, p) => sum + p.correct, 0) / valid.length) * 100;
    }

    getDetailedStats(tableName) {
        const valid = this.predictions[tableName]?.filter(p => p.correct !== null) || [];
        if (!valid.length) return { precision: 0, recall: 0, f1: 0, brier: 0, totalPredictions: 0 };

        let tp = 0, fp = 0, tn = 0, fn = 0, brierSum = 0;
        valid.forEach(p => {
            const actual = p.actualResult;
            const predicted = p.ket_qua_du_doan;
            const prob = p.ti_le_tai / 100;
            const actualBinary = actual === 'tài' ? 1 : 0;
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
            brier,
            totalPredictions: valid.length
        };
    }

    getModelRecentAccuracy(modelName, limit = 20) {
        const accs = this.modelRecentAccuracy[modelName] || [];
        if (!accs.length) return 0.5;
        const recent = accs.slice(-limit);
        return recent.reduce((a, b) => a + b, 0) / recent.length;
    }

    updateModelAccuracy(modelName, correct) {
        if (!this.modelRecentAccuracy[modelName]) this.modelRecentAccuracy[modelName] = [];
        this.modelRecentAccuracy[modelName].push(correct ? 1 : 0);
        if (this.modelRecentAccuracy[modelName].length > 100)
            this.modelRecentAccuracy[modelName] = this.modelRecentAccuracy[modelName].slice(-100);
    }
}

// ==================== PREDICTOR ====================
class Predictor {
    constructor() {
        this.modelNames = ['markov', 'logistic', 'naiveBayes', 'neuralNet', 'legacyPython'];
        this.modelNames.forEach(name => accuracyTracker.registerModel(name));
        this.weights = {};
        this.modelNames.forEach(name => { this.weights[name] = 1.0; });

        this.models = {
            logistic: new LogisticRegression({ numSteps: 1000, learningRate: 0.1 }),
            neuralNet: new brain.NeuralNetwork({ hiddenLayers: [5, 3] })
        };

        this.trainingData = {
            logistic: { features: [], labels: [] },
            neuralNet: { inputs: [], outputs: [] }
        };
        this.minTrainSamples = 20;
    }

    extractFeatures(history, targetSid = null) {
        let relevantHistory = history;
        if (targetSid) relevantHistory = history.filter(s => s.sid < targetSid);
        if (relevantHistory.length === 0) return null;

        const last5 = relevantHistory.slice(-5);
        const last10 = relevantHistory.slice(-10);
        const last20 = relevantHistory.slice(-20);

        const taiRate = (arr) => arr.filter(s => s.ket_qua === 'tài').length / arr.length;

        const features = [
            taiRate(last5),
            taiRate(last10),
            taiRate(last20),
            simpleStats.mean(last10.map(s => s.tong)) || 0,
            simpleStats.standardDeviation(last10.map(s => s.tong)) || 0,
        ];

        for (let i = 1; i <= 5; i++) {
            const idx = relevantHistory.length - i;
            features.push(idx >= 0 ? (relevantHistory[idx].ket_qua === 'tài' ? 1 : 0) : 0.5);
        }

        return features;
    }

    markovPredict(history) {
        if (history.length < 3) return 0.5;
        const seq = history.map(s => s.ket_qua === 'tài' ? 1 : 0);
        const n = seq.length;

        const weight = (idx) => Math.exp((idx - n) / 10);
        let trans1 = { '0->0': 0, '0->1': 0, '1->0': 0, '1->1': 0 };
        let trans2 = {
            '00->0':0, '00->1':0, '01->0':0, '01->1':0,
            '10->0':0, '10->1':0, '11->0':0, '11->1':0
        };

        for (let i = 0; i < n-1; i++) {
            const w = weight(i);
            trans1[`${seq[i]}->${seq[i+1]}`] += w;
        }
        for (let i = 0; i < n-2; i++) {
            const w = weight(i);
            trans2[`${seq[i]}${seq[i+1]}->${seq[i+2]}`] += w;
        }

        const last = seq[n-1];
        const last2 = seq.slice(-2).join('');

        const total1 = trans1[`${last}->0`] + trans1[`${last}->1`];
        const prob1 = total1 === 0 ? 0.5 : trans1[`${last}->1`] / total1;

        const total2 = trans2[`${last2}->0`] + trans2[`${last2}->1`];
        const prob2 = total2 === 0 ? 0.5 : trans2[`${last2}->1`] / total2;

        return (prob1 + 2 * prob2) / 3;
    }

    logisticPredict(history) {
        if (history.length < this.minTrainSamples) return 0.5;
        const X = [], y = [];
        for (let i = 10; i < history.length; i++) {
            const histWindow = history.slice(0, i);
            const features = this.extractFeatures(histWindow, history[i].sid);
            if (features) {
                X.push(features);
                y.push(history[i].ket_qua === 'tài' ? 1 : 0);
            }
        }
        if (X.length < 5) return 0.5;
        try {
            this.models.logistic.train(X, y);
            const lastFeatures = this.extractFeatures(history);
            if (!lastFeatures) return 0.5;
            return this.models.logistic.predictProbabilities([lastFeatures])[0][1];
        } catch (e) {
            console.error('Logistic error:', e.message);
            return 0.5;
        }
    }

    naiveBayesPredict(history) {
        if (history.length < 10) return 0.5;
        const last10 = history.slice(-10);
        return last10.filter(s => s.ket_qua === 'tài').length / 10;
    }

    neuralNetPredict(history) {
        if (history.length < this.minTrainSamples) return 0.5;
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
            return this.models.neuralNet.run(lastFeatures)[0];
        } catch (e) {
            console.error('NeuralNet error:', e.message);
            return 0.5;
        }
    }

    legacyPythonPredict(history) {
        if (history.length < 5) return 0.5;
        const last3 = history.slice(-3);
        const allTai = last3.every(s => s.ket_qua === 'tài');
        const allXiu = last3.every(s => s.ket_qua === 'xỉu');
        if (allTai) return 0.3;
        if (allXiu) return 0.7;
        const last10 = history.slice(-10);
        return last10.filter(s => s.ket_qua === 'tài').length / 10;
    }

    predict(history, tableName) {
        if (!history || history.length < 10) {
            return { error: 'Chưa đủ dữ liệu để dự đoán (cần tối thiểu 10 phiên)' };
        }

        const probs = {};
        this.modelNames.forEach(name => {
            switch (name) {
                case 'markov': probs[name] = this.markovPredict(history); break;
                case 'logistic': probs[name] = this.logisticPredict(history); break;
                case 'naiveBayes': probs[name] = this.naiveBayesPredict(history); break;
                case 'neuralNet': probs[name] = this.neuralNetPredict(history); break;
                case 'legacyPython': probs[name] = this.legacyPythonPredict(history); break;
                default: probs[name] = 0.5;
            }
        });

        let totalWeight = 0;
        const dynamicWeights = {};
        this.modelNames.forEach(name => {
            const recentAcc = accuracyTracker.getModelRecentAccuracy(name, 20);
            const weight = Math.max(0.1, recentAcc);
            dynamicWeights[name] = weight;
            totalWeight += weight;
        });

        let ensembleProb = 0;
        this.modelNames.forEach(name => ensembleProb += probs[name] * dynamicWeights[name]);
        ensembleProb /= totalWeight;

        const ketQua = ensembleProb >= 0.5 ? 'tài' : 'xỉu';
        const tiLeTai = ensembleProb * 100;
        const tiLeXiu = 100 - tiLeTai;

        let doTinCay = 'thấp';
        if (history.length >= 50) doTinCay = 'cao';
        else if (history.length >= 20) doTinCay = 'trung bình';

        const probValues = Object.values(probs);
        const stdDev = simpleStats.standardDeviation(probValues) || 0;
        if (stdDev < 0.1 && history.length >= 20) doTinCay = 'cao';

        const prediction = {
            ket_qua_du_doan: ketQua,
            ti_le_tai: tiLeTai,
            ti_le_xiu: tiLeXiu,
            phien_du_doan: history[history.length-1].sid + 1,
            so_lieu_su_dung: history.length,
            do_tin_cay: doTinCay,
            chi_tiet_mo_hinh: probs,
            trong_so: dynamicWeights
        };

        accuracyTracker.addPrediction(tableName, prediction);
        return prediction;
    }

    updateActual(tableName, sid, actualResult) {
        accuracyTracker.recordActual(tableName, sid, actualResult);
        // Có thể cập nhật thêm độ chính xác cho từng mô hình nếu lưu riêng
    }
}

// ==================== WEBSOCKET CLIENT & EXPRESS SERVER ====================
class GameWebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000;
        this.isAuthenticated = false;
        this.sessionId = null;
        this.latestTxData = null;
        this.latestMd5Data = null;
        this.lastUpdateTime = { tx: null, md5: null };
        this.lastProcessedSid = { tx: 0, md5: 0 };
        this.lastPrediction = { tx: null, md5: null };
    }

    connect() {
        console.log('🔗 Connecting to WebSocket server...');
        this.ws = new WebSocket(this.url, {
            headers: {
                'Host': 'api.jiusyss.me',
                'Origin': 'https://play.son789.site',
                'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
                'Sec-WebSocket-Extensions': 'permessage-deflate; client_max_window_bits',
                'Sec-WebSocket-Version': '13'
            }
        });
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.ws.on('open', () => {
            console.log('✅ Connected to WebSocket server');
            this.reconnectAttempts = 0;
            this.sendAuthentication();
        });

        this.ws.on('message', (data) => this.handleMessage(data));
        this.ws.on('error', (error) => console.error('❌ WebSocket error:', error.message));
        this.ws.on('close', (code, reason) => {
            console.log(`🔌 Connection closed. Code: ${code}, Reason: ${String(reason)}`);
            this.isAuthenticated = false;
            this.sessionId = null;
            this.handleReconnect();
        });
        this.ws.on('pong', () => console.log('❤️ Heartbeat received from server'));
    }

    sendAuthentication() {
        console.log('🔐 Sending authentication...');
        const authMessage = [
            1,
            "MiniGame",
            "son789apia",
            "WangLin1@",
            {
                "signature": "3B807F3D9780682F163184B42F8A3B30B26814FF23F1B7784F99DC842AC076F758E4718F533AF9405F1129E3830A236DAAA0127F1EECA73BC6EB057B5174E4509D57408CCF2C7E316136F98CE46843E6920130C60465D474CABAF6F911E7068DE9B20198CFF684DE6270C9E42922A46E46F5D60EC2BAA9B75F9BE8605E824CA0",
                "info": {
                    "cs": "9e05a39a8958d83119db6ab9a1d88548",
                    "phone": "",
                    "ipAddress": "113.185.46.68",
                    "isMerchant": false,
                    "userId": "bf5dc66b-2e77-4b48-ab73-09f2ffbe3443",
                    "deviceId": "050105373613900053736078036024",
                    "isMktAccount": false,
                    "username": "son789apia",
                    "timestamp": 1766557267829
                },
                "pid": 4
            }
        ];
        this.sendRaw(authMessage);
    }

    sendPluginMessages() {
        console.log('🚀 Sending plugin initialization messages...');
        const pluginMessages = [
            [6, "MiniGame", "taixiuPlugin", { "cmd": 1005 }],
            [6, "MiniGame", "taixiuMd5Plugin", { "cmd": 1105 }],
            [6, "MiniGame", "lobbyPlugin", { "cmd": 10001 }],
            [6, "MiniGame", "channelPlugin", { "cmd": 310 }]
        ];
        pluginMessages.forEach((message, index) => {
            setTimeout(() => {
                console.log(`📤 Sending plugin ${index + 1}/${pluginMessages.length}: ${message[2]}`);
                this.sendRaw(message);
            }, index * 1000);
        });
        setInterval(() => this.refreshGameData(), 30000);
    }

    refreshGameData() {
        if (this.isAuthenticated && this.ws?.readyState === WebSocket.OPEN) {
            console.log('🔄 Refreshing game data...');
            this.sendRaw([6, "MiniGame", "taixiuPlugin", { "cmd": 1005 }]);
            setTimeout(() => this.sendRaw([6, "MiniGame", "taixiuMd5Plugin", { "cmd": 1105 }]), 1000);
        }
    }

    sendRaw(data) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            const jsonString = JSON.stringify(data);
            this.ws.send(jsonString);
            console.log('📤 Sent raw:', jsonString);
            return true;
        }
        console.log('⚠️ Cannot send, WebSocket not open');
        return false;
    }

    handleMessage(data) {
        try {
            const parsed = JSON.parse(data);
            if (parsed[0] === 5 && parsed[1]?.cmd === 1005) {
                console.log('🎯 Nhận được dữ liệu cmd 1005 (Bàn TX)');
                this.processGameData('tx', parsed[1]);
            } else if (parsed[0] === 5 && parsed[1]?.cmd === 1105) {
                console.log('🎯 Nhận được dữ liệu cmd 1105 (Bàn MD5)');
                this.processGameData('md5', parsed[1]);
            } else if (parsed[0] === 5 && parsed[1]?.cmd === 100) {
                console.log('🔑 Authentication successful!');
                console.log(`✅ User: ${parsed[1].u}`);
                this.isAuthenticated = true;
                setTimeout(() => {
                    console.log('🔄 Starting to send plugin messages...');
                    this.sendPluginMessages();
                }, 2000);
            } else if (parsed[0] === 1 && parsed.length >= 5 && parsed[4] === "MiniGame") {
                console.log('✅ Session initialized');
                this.sessionId = parsed[3];
                console.log(`📋 Session ID: ${this.sessionId}`);
            } else if (parsed[0] === 7) {
                console.log(`🔄 Plugin ${parsed[2]} response received`);
            } else if (parsed[0] === 0) {
                console.log('❤️ Heartbeat received');
            }
        } catch (e) {
            console.log('📥 Raw message:', data.toString());
            console.error('❌ Parse error:', e.message);
        }
    }

    processGameData(tableName, gameData) {
        if (!gameData.htr || !Array.isArray(gameData.htr) || gameData.htr.length === 0) return;
        if (tableName === 'tx') {
            this.latestTxData = gameData;
            this.lastUpdateTime.tx = new Date();
        } else {
            this.latestMd5Data = gameData;
            this.lastUpdateTime.md5 = new Date();
        }

        gameData.htr.forEach(session => {
            const tong = session.d1 + session.d2 + session.d3;
            const ketQua = tong >= 11 ? 'tài' : 'xỉu';
            const newSession = {
                sid: session.sid,
                d1: session.d1,
                d2: session.d2,
                d3: session.d3,
                tong: tong,
                ket_qua: ketQua,
                thoi_gian: new Date().toISOString()
            };

            const added = historyManager.addSession(tableName, newSession);
            if (added && session.sid > this.lastProcessedSid[tableName]) {
                this.lastProcessedSid[tableName] = session.sid;
                const history = historyManager.getHistory(tableName);
                if (history.length >= 10) {
                    setTimeout(() => {
                        const prediction = predictor.predict(history, tableName);
                        if (!prediction.error) {
                            this.lastPrediction[tableName] = prediction;
                            console.log(`🔮 Dự đoán bàn ${tableName} - phiên ${prediction.phien_du_doan}: ${prediction.ket_qua_du_doan} (Tài: ${prediction.ti_le_tai.toFixed(2)}%)`);
                        }
                    }, 0);
                }

                const predictions = accuracyTracker.predictions[tableName] || [];
                const matchingPred = predictions.find(p => p.phien_du_doan === session.sid);
                if (matchingPred && matchingPred.actualResult === null) {
                    predictor.updateActual(tableName, session.sid, ketQua);
                    console.log(`✅ Cập nhật kết quả thực tế bàn ${tableName} - phiên ${session.sid}: ${ketQua}`);
                }
            }
        });
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * this.reconnectAttempts;
            console.log(`🔄 Attempting to reconnect in ${delay}ms (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
                console.log('🔄 Reconnecting...');
                this.connect();
            }, delay);
        } else {
            console.log('❌ Max reconnection attempts reached');
        }
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.sendRaw([0, this.sessionId || ""]);
                console.log('❤️ Sending heartbeat...');
            }
        }, 25000);
    }

    close() {
        this.ws?.close();
    }
}

// ==================== KHỞI TẠO CÁC ĐỐI TƯỢNG TOÀN CỤC ====================
const historyManager = new HistoryManager();
const accuracyTracker = new AccuracyTracker();
const predictor = new Predictor();

// ==================== EXPRESS SERVER ====================
const app = express();
const PORT = process.env.PORT || 3000;
app.use(cors());
app.use(express.json());

const client = new GameWebSocketClient(
    'wss://api.jiusyss.me/websocket?d=YUd0aGIyNWliMmM9fDEyOTh8MTc2NjU1NzI2NzI2M3wzMmNlMmE1NGQzNmFhY2FhMWZmNjZmMzE5MzQ1ZmUyNXw5MjJjMjBhMTE4NTBiNzRiNmNjYzQxMTE3Nzk0NDQ5Zg=='
);
client.connect();

// ==================== API ENDPOINTS ====================
app.get('/api/tx', (req, res) => {
    const latest = historyManager.getLatestSession('tx');
    if (!latest) return res.status(404).json({ error: 'Không có dữ liệu bàn TX' });
    res.json({ ...latest, ban: 'tai_xiu', last_updated: client.lastUpdateTime.tx });
});

app.get('/api/md5', (req, res) => {
    const latest = historyManager.getLatestSession('md5');
    if (!latest) return res.status(404).json({ error: 'Không có dữ liệu bàn MD5' });
    res.json({ ...latest, ban: 'md5', last_updated: client.lastUpdateTime.md5 });
});

app.get('/api/all', (req, res) => {
    res.json({
        tai_xiu: historyManager.getLatestSession('tx') || { error: 'Không có dữ liệu' },
        md5: historyManager.getLatestSession('md5') || { error: 'Không có dữ liệu' },
        timestamp: new Date().toISOString()
    });
});

app.get('/api/predict/tx', (req, res) => {
    if (!client.lastPrediction.tx) return res.status(400).json({ error: 'Chưa có dự đoán cho bàn TX' });
    res.json(client.lastPrediction.tx);
});

app.get('/api/predict/md5', (req, res) => {
    if (!client.lastPrediction.md5) return res.status(400).json({ error: 'Chưa có dự đoán cho bàn MD5' });
    res.json(client.lastPrediction.md5);
});

app.get('/api/predict/all', (req, res) => {
    res.json({
        tx: client.lastPrediction.tx || { error: 'Chưa có dự đoán TX' },
        md5: client.lastPrediction.md5 || { error: 'Chưa có dự đoán MD5' }
    });
});

app.get('/api/accuracy', (req, res) => {
    res.json({
        tx: {
            overall_accuracy: accuracyTracker.getOverallAccuracy('tx'),
            recent_50_accuracy: accuracyTracker.getRecentAccuracy('tx', 50),
            ...accuracyTracker.getDetailedStats('tx')
        },
        md5: {
            overall_accuracy: accuracyTracker.getOverallAccuracy('md5'),
            recent_50_accuracy: accuracyTracker.getRecentAccuracy('md5', 50),
            ...accuracyTracker.getDetailedStats('md5')
        }
    });
});

app.get('/api/history/:ban', (req, res) => {
    const ban = req.params.ban;
    if (!['tx', 'md5'].includes(ban)) return res.status(400).json({ error: 'Bàn không hợp lệ' });
    res.json(historyManager.getHistory(ban, 100));
});

app.get('/api/stats/:ban', (req, res) => {
    const ban = req.params.ban;
    if (!['tx', 'md5'].includes(ban)) return res.status(400).json({ error: 'Bàn không hợp lệ' });
    const history = historyManager.getHistory(ban);
    if (!history.length) return res.json({ message: 'Chưa có dữ liệu', total: 0 });

    const total = history.length;
    const taiCount = history.filter(s => s.ket_qua === 'tài').length;
    const tiLeTai = (taiCount / total) * 100;
    const tongDiem = history.map(s => s.tong);
    const trungBinh = simpleStats.mean(tongDiem);
    const doLechChuan = simpleStats.standardDeviation(tongDiem) || 0;

    const phanBo = {};
    for (let i = 3; i <= 18; i++) phanBo[i] = history.filter(s => s.tong === i).length;

    res.json({
        tong_so_phien: total,
        ti_le_tai: tiLeTai,
        ti_le_xiu: 100 - tiLeTai,
        trung_binh_tong: trungBinh,
        do_lech_chuan_tong: doLechChuan,
        phan_bo_tong_diem: phanBo
    });
});

app.get('/api/status', (req, res) => {
    res.json({
        status: "running",
        websocket_connected: client.ws?.readyState === WebSocket.OPEN,
        authenticated: client.isAuthenticated,
        has_tx_data: historyManager.getHistory('tx').length > 0,
        has_md5_data: historyManager.getHistory('md5').length > 0,
        tx_last_updated: client.lastUpdateTime.tx,
        md5_last_updated: client.lastUpdateTime.md5,
        tx_history_size: historyManager.getHistory('tx').length,
        md5_history_size: historyManager.getHistory('md5').length,
        timestamp: new Date().toISOString()
    });
});

app.get('/api/refresh', (req, res) => {
    if (client.isAuthenticated && client.ws?.readyState === WebSocket.OPEN) {
        client.refreshGameData();
        res.json({ message: "Đã gửi yêu cầu refresh dữ liệu", timestamp: new Date().toISOString() });
    } else {
        res.status(400).json({ error: "Không thể refresh" });
    }
});

app.get('/', (req, res) => {
    res.send(`
        <html>
            <head><title>API Dự đoán Tài Xỉu</title></head>
            <body>
                <h1>API Dự đoán Tài Xỉu</h1>
                <p>Các endpoints:</p>
                <ul>
                    <li>/api/tx - Phiên mới nhất bàn TX</li>
                    <li>/api/md5 - Phiên mới nhất bàn MD5</li>
                    <li>/api/all - Cả hai bàn</li>
                    <li>/api/predict/tx - Dự đoán bàn TX</li>
                    <li>/api/predict/md5 - Dự đoán bàn MD5</li>
                    <li>/api/predict/all - Dự đoán cả hai</li>
                    <li>/api/accuracy - Độ chính xác</li>
                    <li>/api/history/:ban - Lịch sử phiên (tx/md5)</li>
                    <li>/api/stats/:ban - Thống kê bàn</li>
                    <li>/api/status - Trạng thái hệ thống</li>
                </ul>
            </body>
        </html>
    `);
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Server đang chạy tại: http://localhost:${PORT}`);
});

setTimeout(() => client.startHeartbeat(), 10000);

process.on('SIGINT', () => {
    console.log('\n👋 Closing WebSocket connection and server...');
    client.close();
    process.exit();
});
