// ============================================================
// CardioSound AI — Gestor de Audio
// Maneja grabación por micrófono, carga de archivos y remuestreo
// ============================================================

/**
 * Clase para gestionar la entrada de audio:
 * - Grabación desde micrófono con buffer circular de 3 segundos
 * - Carga y decodificación de archivos de audio
 * - Remuestreo a la frecuencia objetivo (22050 Hz)
 */
export class AudioManager {
    /**
     * @param {object} config - Configuración compartida del pipeline de audio
     */
    constructor(config) {
        // Configuración del pipeline
        this.sampleRate = config.SAMPLE_RATE || 22050;      // 22050
        this.duration = config.DURATION || 3.0;            // 3 segundos
        this.nSamples = config.N_SAMPLES || 66150;           // 66150

        // Estado del micrófono
        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.analyserNode = null;
        this.processorNode = null;

        // Buffer dinámico para almacenar muestras del micrófono
        this.recordedChunks = [];
        this.samplesCollected = 0;
        this.isRecording = false;

        // Sistema de eventos sencillo
        this._listeners = {};

        // Para reproducción de muestras
        this._playbackContext = null;
        this._playbackSource = null;
    }

    // ── Sistema de eventos ──

    /**
     * Registrar un listener para un evento
     * @param {string} event - Nombre del evento
     * @param {Function} callback - Función a ejecutar
     */
    on(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
    }

    /**
     * Emitir un evento a todos los listeners registrados
     * @param {string} event - Nombre del evento
     * @param {*} data - Datos del evento
     */
    _emit(event, data) {
        if (this._listeners[event]) {
            this._listeners[event].forEach(cb => cb(data));
        }
    }

    // ── Micrófono ──

    /**
     * Iniciar captura desde el micrófono
     * Configura AudioContext a 22050 Hz, AnalyserNode y ScriptProcessor
     * para recopilar muestras en un buffer circular de 3 segundos
     */
    async startMicrophone(monitorOnly = false, inputDeviceId = null, outputDeviceId = null) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Tu navegador no soporta grabación o requiere estar en un entorno seguro (HTTPS).');
        }

        try {
            const audioConstraints = {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            };
            
            if (inputDeviceId && inputDeviceId !== 'default') {
                audioConstraints.deviceId = { exact: inputDeviceId };
            }

            // Solicitar acceso al micrófono sin procesamiento de audio
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: audioConstraints
            });

            // Crear AudioContext con la frecuencia de muestreo objetivo (solo usado para grabar y procesar)
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            // Requisito para Safari en iOS: reanudar el contexto explícitamente
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Nodo fuente desde el stream del micrófono
            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

            // AnalyserNode para visualización en tiempo real
            this.analyserNode = this.audioContext.createAnalyser();
            this.analyserNode.fftSize = 2048;
            this.analyserNode.smoothingTimeConstant = 0.8;

            this.sourceNode.connect(this.analyserNode);

            if (monitorOnly) {
                // Modo Escucha: usar HTMLAudioElement para mayor compatibilidad con setSinkId
                this._monitorAudioElement = new Audio();
                this._monitorAudioElement.srcObject = this.mediaStream;
                if (outputDeviceId && outputDeviceId !== 'default' && typeof this._monitorAudioElement.setSinkId === 'function') {
                    try {
                        await this._monitorAudioElement.setSinkId(outputDeviceId);
                    } catch (e) {
                        console.warn('[AudioManager] HTMLAudioElement setSinkId falló:', e);
                    }
                }
                
                // Forzar altavoz de forma nativa en Android
                if (window.CardioAndroid) {
                    try {
                        window.CardioAndroid.setSpeaker(true);
                    } catch (e) {}
                }
                
                try {
                    await this._monitorAudioElement.play();
                } catch (e) {
                    console.error("[AudioManager] No se pudo iniciar el monitor de audio:", e);
                }

                this.isRecording = false;
                this.isMonitoring = true;
                this._emit('monitoringStarted');
            } else {
                // Modo Grabación
                // ScriptProcessor para capturar muestras crudas (deprecated pero universalmente soportado)
                const bufferSize = 4096;
                this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

                // Resetear buffer dinámico
                this.recordedChunks = [];
                this.samplesCollected = 0;
                this.isRecording = true;
                this.isMonitoring = false;

                // Procesar cada bloque de audio entrante
                this.processorNode.onaudioprocess = (e) => {
                    if (!this.isRecording) return;

                    const inputData = e.inputBuffer.getChannelData(0);

                    // Escribir muestras en el buffer dinámico
                    this.recordedChunks.push(new Float32Array(inputData));
                    this.samplesCollected += inputData.length;

                    // Emitir progreso (ahora puede pasar del 100%)
                    const progress = this.samplesCollected / this.nSamples;
                    this._emit('recordingProgress', progress);
                };

                // Conectar la cadena de nodos: analyser → processor → destination
                this.analyserNode.connect(this.processorNode);
                this.processorNode.connect(this.audioContext.destination);

                this._emit('recordingStarted');
            }

        } catch (error) {
            console.error('[AudioManager] Error al iniciar micrófono:', error);
            this._emit('error', {
                type: 'microphone',
                message: error.name === 'NotAllowedError'
                    ? 'Permiso de micrófono denegado. Por favor, permite el acceso al micrófono.'
                    : `Error al acceder al micrófono: ${error.message}`
            });
            throw error;
        }
    }

    /**
     * Detener la captura del micrófono y liberar recursos
     */
    stopMicrophone() {
        this.isRecording = false;
        this.isMonitoring = false;

        if (this._monitorAudioElement) {
            try { this._monitorAudioElement.pause(); } catch (e) {}
            this._monitorAudioElement.srcObject = null;
            this._monitorAudioElement = null;
        }

        // Desconectar nodos
        if (this.processorNode) {
            this.processorNode.onaudioprocess = null;
            try { this.processorNode.disconnect(); } catch (e) { /* ignorar */ }
            this.processorNode = null;
        }
        if (this.analyserNode) {
            try { this.analyserNode.disconnect(); } catch (e) { /* ignorar */ }
        }
        if (this.sourceNode) {
            try { this.sourceNode.disconnect(); } catch (e) { /* ignorar */ }
            this.sourceNode = null;
        }

        // Detener todas las pistas del stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Cerrar AudioContext
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => { /* ignorar */ });
            this.audioContext = null;
        }

        this._emit('recordingStopped');
    }

    /**
     * Obtener datos de frecuencia del AnalyserNode para visualización
     * @returns {Uint8Array|null} Datos de frecuencia o null
     */
    getAnalyserData() {
        if (!this.analyserNode) return null;
        const data = new Uint8Array(this.analyserNode.frequencyBinCount);
        this.analyserNode.getByteFrequencyData(data);
        return data;
    }

    /**
     * Obtener el buffer de audio completo grabado
     * Concatena todos los chunks recopilados
     * @returns {Float32Array} Buffer de audio completo
     */
    getAudioBuffer() {
        const result = new Float32Array(this.samplesCollected);
        let offset = 0;
        for (const chunk of this.recordedChunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    // ── Carga de archivos ──

    /**
     * Cargar un archivo de audio desde un File object
     * Decodifica y remuestrea a 22050 Hz
     * @param {File} file - Archivo de audio
     * @returns {Promise<Float32Array>} Buffer de audio remuestreado
     */
    async loadAudioFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = async (e) => {
                try {
                    const audioData = await this._decodeAndResample(e.target.result);
                    resolve(audioData);
                } catch (err) {
                    reject(err);
                }
            };

            reader.onerror = () => reject(new Error('Error al leer el archivo'));
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Cargar audio desde una URL
     * @param {string} url - URL del archivo de audio
     * @returns {Promise<Float32Array>} Buffer de audio remuestreado
     */
    async loadAudioUrl(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            const arrayBuffer = await response.arrayBuffer();
            return await this._decodeAndResample(arrayBuffer);
        } catch (error) {
            console.error('[AudioManager] Error al cargar URL:', error);
            throw new Error(`No se pudo cargar el audio: ${error.message}`);
        }
    }

    /**
     * Decodificar un ArrayBuffer de audio y remuestrear a la frecuencia objetivo
     * Recorta o rellena a exactamente N_SAMPLES muestras
     * @param {ArrayBuffer} arrayBuffer - Datos de audio codificados
     * @returns {Promise<Float32Array>} Buffer de audio remuestreado
     * @private
     */
    async _decodeAndResample(arrayBuffer) {
        // Crear AudioContext temporal para decodificar
        const tempCtx = new (window.AudioContext || window.webkitAudioContext)();

        try {
            const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);

            // Remuestrear usando OfflineAudioContext
            const numFrames = Math.ceil(audioBuffer.duration * this.sampleRate);
            const offlineCtx = new OfflineAudioContext(1, numFrames, this.sampleRate);
            const source = offlineCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineCtx.destination);
            source.start(0);

            const renderedBuffer = await offlineCtx.startRendering();
            const channelData = renderedBuffer.getChannelData(0);

            // Devolver el audio completo remuestreado
            return channelData;

        } finally {
            // Cerrar contexto temporal
            if (tempCtx.state !== 'closed') {
                tempCtx.close().catch(() => { /* ignorar */ });
            }
        }
    }

    // ── Reproducción de muestras ──

    /**
     * Reproducir un archivo de audio desde una URL
     * @param {string} url - URL del archivo de audio
     */
    async playSample(url) {
        this.stopPlayback();

        try {
            this._playbackContext = new (window.AudioContext || window.webkitAudioContext)();
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await this._playbackContext.decodeAudioData(arrayBuffer);

            this._playbackSource = this._playbackContext.createBufferSource();
            this._playbackSource.buffer = audioBuffer;
            this._playbackSource.connect(this._playbackContext.destination);
            this._playbackSource.start(0);

            this._playbackSource.onended = () => {
                this.stopPlayback();
            };
        } catch (error) {
            console.error('[AudioManager] Error al reproducir:', error);
            this.stopPlayback();
        }
    }

    /**
     * Detener la reproducción actual
     */
    stopPlayback() {
        if (this._playbackAudioElement) {
            try { this._playbackAudioElement.pause(); } catch (e) { /* ignorar */ }
            this._playbackAudioElement.onended = null;
            this._playbackAudioElement.src = '';
            this._playbackAudioElement = null;
        }
        if (this._playbackUrl) {
            URL.revokeObjectURL(this._playbackUrl);
            this._playbackUrl = null;
        }

        if (this._playbackSource) {
            try { this._playbackSource.stop(); } catch (e) { /* ignorar */ }
            this._playbackSource = null;
        }
        if (this._playbackContext && this._playbackContext.state !== 'closed') {
            this._playbackContext.close().catch(() => { /* ignorar */ });
            this._playbackContext = null;
        }
    }

    /**
     * Exporta el buffer grabado a formato WAV
     * @returns {Blob|null} Blob de audio/wav o null si no hay datos
     */
    exportWAV() {
        const samples = this.getAudioBuffer();
        if (!samples || samples.length === 0) return null;

        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        
        // fmt sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, this.sampleRate, true);
        view.setUint32(28, this.sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        
        // data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        // write PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    /**
     * Reproduce el buffer de audio grabado
     * @param {Function} onEndedCallback - Callback cuando termina la reproducción
     * @param {string} outputDeviceId - ID del dispositivo de salida
     * @returns {Promise<boolean>} Si pudo iniciar la reproducción
     */
    async playRecordedAudio(onEndedCallback, outputDeviceId = null) {
        this.stopPlayback();
        const samples = this.getAudioBuffer();
        if (!samples || samples.length === 0) return false;

        try {
            // Generar WAV para usar HTMLAudioElement, más compatible con setSinkId
            const wavBlob = this.exportWAV();
            if (!wavBlob) return false;

            this._playbackUrl = URL.createObjectURL(wavBlob);
            this._playbackAudioElement = new Audio(this._playbackUrl);
            
            // Configurar salida si es posible
            if (outputDeviceId && outputDeviceId !== 'default' && typeof this._playbackAudioElement.setSinkId === 'function') {
                try {
                    await this._playbackAudioElement.setSinkId(outputDeviceId);
                } catch (e) {
                    console.warn('[AudioManager] HTMLAudioElement setSinkId falló:', e);
                }
            }
            
            // Forzar altavoz de forma nativa en Android si el plugin está disponible
            if (window.CardioAndroid) {
                try {
                    window.CardioAndroid.setSpeaker(true);
                } catch (e) {
                    console.warn('[AudioManager] Native CardioAndroid falló:', e);
                }
            }

            this._playbackAudioElement.onended = () => {
                this.stopPlayback();
                if (onEndedCallback) onEndedCallback();
            };
            
            await this._playbackAudioElement.play();
            return true;
        } catch (error) {
            console.error('[AudioManager] Error al reproducir audio grabado:', error);
            this.stopPlayback();
            return false;
        }
    }

    /**
     * Liberar todos los recursos
     */
    dispose() {
        this.stopMicrophone();
        this.stopPlayback();
        this._listeners = {};
    }
}
