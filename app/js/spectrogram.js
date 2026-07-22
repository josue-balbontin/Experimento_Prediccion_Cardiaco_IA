// ============================================================
// CardioSound AI — Motor de Espectrograma Mel
// Cálculo de espectrograma compatible con librosa y visualización
// ============================================================

/**
 * Motor de espectrograma Mel que replica el pipeline de librosa:
 * - STFT con ventana Hann
 * - Banco de filtros Mel triangulares
 * - Conversión a escala de dB
 * - Visualización con colormap magma
 */
export class SpectrogramEngine {
    /**
     * @param {object} config - Configuración compartida del pipeline
     */
    constructor(config) {
        this.sampleRate = config.SAMPLE_RATE;   // 22050
        this.nFft = config.N_FFT;               // 2048
        this.hopLength = config.HOP_LENGTH;     // 512
        this.nMels = config.N_MELS;             // 128
        this.fmin = config.FMIN;                // 20
        this.fmax = config.FMAX;                // 4000
        this.imgSize = config.IMG_SIZE;          // 224

        // Pre-calcular la ventana Hann y el banco de filtros Mel
        this._hannWindow = this._hann(this.nFft);
        this._melFilterbank = this._createMelFilterbank();

        // Estado de visualización en tiempo real
        this._realtimeAnimId = null;
        this._realtimeColumns = [];
    }

    // ════════════════════════════════════════════════════════════
    // CONVERSIONES MEL ↔ HZ
    // ════════════════════════════════════════════════════════════

    /**
     * Convertir frecuencia en Hz a escala Mel (fórmula de Slaney/HTK)
     * @param {number} hz - Frecuencia en Hertz
     * @returns {number} Valor en escala Mel
     */
    _hzToMel(hz) {
        return 2595.0 * Math.log10(1.0 + hz / 700.0);
    }

    /**
     * Convertir escala Mel a frecuencia en Hz
     * @param {number} mel - Valor en escala Mel
     * @returns {number} Frecuencia en Hertz
     */
    _melToHz(mel) {
        return 700.0 * (Math.pow(10, mel / 2595.0) - 1.0);
    }

    // ════════════════════════════════════════════════════════════
    // BANCO DE FILTROS MEL
    // ════════════════════════════════════════════════════════════

    /**
     * Crear banco de filtros Mel triangulares [nMels × (nFft/2+1)]
     * Replica librosa.filters.mel con norma 'slaney'
     * @returns {Float64Array[]} Matriz de filtros Mel
     */
    _createMelFilterbank() {
        const numFreqBins = Math.floor(this.nFft / 2) + 1; // 1025
        const numFilterPoints = this.nMels + 2;             // 130

        // Puntos Mel espaciados linealmente
        const melMin = this._hzToMel(this.fmin);
        const melMax = this._hzToMel(this.fmax);
        const melPoints = new Float64Array(numFilterPoints);
        for (let i = 0; i < numFilterPoints; i++) {
            melPoints[i] = melMin + (melMax - melMin) * i / (numFilterPoints - 1);
        }

        // Convertir puntos Mel a Hz y luego a índices de bins FFT
        const hzPoints = melPoints.map(m => this._melToHz(m));
        const binPoints = hzPoints.map(hz =>
            Math.floor((this.nFft + 1) * hz / this.sampleRate)
        );

        // Crear filtros triangulares
        const filterbank = [];
        for (let m = 0; m < this.nMels; m++) {
            const filter = new Float64Array(numFreqBins);
            const startBin = binPoints[m];
            const centerBin = binPoints[m + 1];
            const endBin = binPoints[m + 2];

            // Rampa ascendente: de startBin a centerBin
            for (let k = startBin; k <= centerBin; k++) {
                if (k >= 0 && k < numFreqBins) {
                    const denom = centerBin - startBin;
                    filter[k] = denom > 0 ? (k - startBin) / denom : 0;
                }
            }

            // Rampa descendente: de centerBin a endBin
            for (let k = centerBin; k <= endBin; k++) {
                if (k >= 0 && k < numFreqBins) {
                    const denom = endBin - centerBin;
                    filter[k] = denom > 0 ? (endBin - k) / denom : 0;
                }
            }

            // Normalización Slaney: normalizar por el ancho del filtro en Hz
            const enorm = 2.0 / (this._melToHz(melPoints[m + 2]) - this._melToHz(melPoints[m]));
            for (let k = 0; k < numFreqBins; k++) {
                filter[k] *= enorm;
            }

            filterbank.push(filter);
        }

        return filterbank;
    }

    // ════════════════════════════════════════════════════════════
    // VENTANA HANN
    // ════════════════════════════════════════════════════════════

    /**
     * Crear ventana Hann (coseno alzado) de longitud n
     * @param {number} n - Tamaño de la ventana
     * @returns {Float64Array} Ventana Hann
     */
    _hann(n) {
        const window = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            window[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / n));
        }
        return window;
    }

    // ════════════════════════════════════════════════════════════
    // FFT (RADIX-2 COOLEY-TUKEY)
    // ════════════════════════════════════════════════════════════

    /**
     * FFT in-place Radix-2 Cooley-Tukey
     * Los arreglos real e imag se modifican directamente
     * @param {Float64Array} real - Parte real
     * @param {Float64Array} imag - Parte imaginaria
     */
    _fft(real, imag) {
        const n = real.length;
        if (n <= 1) return;

        // Reordenamiento bit-reversal
        let j = 0;
        for (let i = 0; i < n; i++) {
            if (i < j) {
                // Intercambiar real
                let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
                // Intercambiar imaginario
                tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
            }
            let m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Mariposas FFT
        for (let size = 2; size <= n; size <<= 1) {
            const halfSize = size >> 1;
            const angle = -2.0 * Math.PI / size;

            for (let i = 0; i < n; i += size) {
                for (let k = 0; k < halfSize; k++) {
                    const theta = angle * k;
                    const wr = Math.cos(theta);
                    const wi = Math.sin(theta);

                    const idx1 = i + k;
                    const idx2 = i + k + halfSize;

                    const tr = wr * real[idx2] - wi * imag[idx2];
                    const ti = wr * imag[idx2] + wi * real[idx2];

                    real[idx2] = real[idx1] - tr;
                    imag[idx2] = imag[idx1] - ti;
                    real[idx1] += tr;
                    imag[idx1] += ti;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════
    // STFT (SHORT-TIME FOURIER TRANSFORM)
    // ════════════════════════════════════════════════════════════

    /**
     * Calcular STFT centrado (como librosa con center=True)
     * Aplica ventana Hann y zero-padding, retorna espectro de potencia
     * @param {Float32Array} audio - Señal de audio mono
     * @returns {Float64Array[]} Espectro de potencia [nFrames × (nFft/2+1)]
     */
    _stft(audio) {
        const numFreqBins = Math.floor(this.nFft / 2) + 1;

        // Padding reflectivo como librosa center=True
        const padLength = Math.floor(this.nFft / 2);
        const paddedLength = audio.length + 2 * padLength;
        const padded = new Float64Array(paddedLength);

        // Relleno reflectivo al inicio
        for (let i = 0; i < padLength; i++) {
            padded[i] = audio[padLength - i] || 0;
        }
        // Datos originales
        for (let i = 0; i < audio.length; i++) {
            padded[padLength + i] = audio[i];
        }
        // Relleno reflectivo al final
        for (let i = 0; i < padLength; i++) {
            const srcIdx = audio.length - 2 - i;
            padded[padLength + audio.length + i] = srcIdx >= 0 ? audio[srcIdx] : 0;
        }

        // Calcular número de frames
        const nFrames = 1 + Math.floor((paddedLength - this.nFft) / this.hopLength);
        const powerSpectrum = [];

        // FFT necesita potencia de 2
        const fftSize = this.nFft; // 2048 ya es potencia de 2

        for (let frame = 0; frame < nFrames; frame++) {
            const start = frame * this.hopLength;
            const real = new Float64Array(fftSize);
            const imag = new Float64Array(fftSize);

            // Aplicar ventana Hann a la trama
            for (let i = 0; i < this.nFft; i++) {
                const sampleIdx = start + i;
                real[i] = (sampleIdx < paddedLength ? padded[sampleIdx] : 0) * this._hannWindow[i];
            }

            // Ejecutar FFT
            this._fft(real, imag);

            // Calcular espectro de potencia (magnitud al cuadrado)
            const power = new Float64Array(numFreqBins);
            for (let k = 0; k < numFreqBins; k++) {
                power[k] = real[k] * real[k] + imag[k] * imag[k];
            }

            powerSpectrum.push(power);
        }

        return powerSpectrum;
    }

    // ════════════════════════════════════════════════════════════
    // CÁLCULO DEL ESPECTROGRAMA MEL
    // ════════════════════════════════════════════════════════════

    /**
     * Calcular espectrograma Mel completo (replica librosa.feature.melspectrogram)
     * Pipeline: STFT → Filtros Mel → Power-to-dB → Normalización [0,1]
     * @param {Float32Array} audioData - Señal de audio (66150 muestras a 22050 Hz)
     * @returns {Float64Array[]} Espectrograma Mel [nMels × nFrames] normalizado [0,1]
     */
    computeMelSpectrogram(audioData) {
        // 1. Calcular STFT (espectro de potencia)
        const powerSpectrum = this._stft(audioData);
        const nFrames = powerSpectrum.length;
        const numFreqBins = Math.floor(this.nFft / 2) + 1;

        // 2. Aplicar banco de filtros Mel
        const melSpec = [];
        for (let m = 0; m < this.nMels; m++) {
            const melRow = new Float64Array(nFrames);
            for (let t = 0; t < nFrames; t++) {
                let sum = 0;
                for (let k = 0; k < numFreqBins; k++) {
                    sum += this._melFilterbank[m][k] * powerSpectrum[t][k];
                }
                melRow[t] = sum;
            }
            melSpec.push(melRow);
        }

        // 3. Convertir a dB: 10 * log10(max(S, 1e-10))
        let globalMax = -Infinity;
        for (let m = 0; m < this.nMels; m++) {
            for (let t = 0; t < nFrames; t++) {
                melSpec[m][t] = 10.0 * Math.log10(Math.max(melSpec[m][t], 1e-10));
                if (melSpec[m][t] > globalMax) globalMax = melSpec[m][t];
            }
        }

        // 4. Normalizar relativo al máximo (como librosa power_to_db con ref=np.max)
        // y luego escalar a [0, 1]
        let globalMin = Infinity;
        for (let m = 0; m < this.nMels; m++) {
            for (let t = 0; t < nFrames; t++) {
                melSpec[m][t] = Math.max(melSpec[m][t], globalMax - 80.0); // top_db = 80
                if (melSpec[m][t] < globalMin) globalMin = melSpec[m][t];
            }
        }

        // Normalización min-max a [0, 1]
        const range = globalMax - globalMin || 1;
        for (let m = 0; m < this.nMels; m++) {
            for (let t = 0; t < nFrames; t++) {
                melSpec[m][t] = (melSpec[m][t] - globalMin) / range;
            }
        }

        return melSpec;
    }

    // ════════════════════════════════════════════════════════════
    // COLORMAP MAGMA
    // ════════════════════════════════════════════════════════════

    /**
     * Colormap Magma para visualización de espectrogramas
     * Interpola entre 25 puntos de control de color
     */
    static MAGMA = [
        [0, 0, 4], [1, 0, 11], [3, 1, 22], [7, 3, 38],
        [16, 7, 54], [29, 11, 69], [44, 15, 81], [60, 18, 90],
        [78, 21, 95], [96, 24, 96], [115, 28, 94], [133, 33, 89],
        [152, 39, 83], [170, 47, 76], [187, 57, 69], [203, 70, 62],
        [218, 86, 56], [230, 104, 52], [240, 125, 51], [247, 148, 53],
        [251, 172, 60], [253, 197, 73], [253, 222, 92], [252, 247, 118],
        [252, 253, 191]
    ];

    /**
     * Obtener color del colormap magma para un valor normalizado [0, 1]
     * @param {number} value - Valor normalizado entre 0 y 1
     * @returns {number[]} [R, G, B] valores 0-255
     */
    _getColor(value) {
        const cmap = SpectrogramEngine.MAGMA;
        const v = Math.max(0, Math.min(1, value));
        const idx = v * (cmap.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, cmap.length - 1);
        const frac = idx - lo;

        return [
            Math.round(cmap[lo][0] + (cmap[hi][0] - cmap[lo][0]) * frac),
            Math.round(cmap[lo][1] + (cmap[hi][1] - cmap[lo][1]) * frac),
            Math.round(cmap[lo][2] + (cmap[hi][2] - cmap[lo][2]) * frac)
        ];
    }

    // ════════════════════════════════════════════════════════════
    // CONVERSIÓN A IMAGEN PARA EL MODELO
    // ════════════════════════════════════════════════════════════

    /**
     * Convertir espectrograma Mel a tensor TF.js [1, 224, 224, 3] para el modelo
     * Redimensiona con interpolación bilineal y usa valores de escala de grises (3 canales iguales)
     * @param {Float64Array[]} melSpec - Espectrograma [nMels × nFrames]
     * @returns {tf.Tensor4D} Tensor [1, 224, 224, 3] listo para inferencia
     */
    spectrogramToTensor(melSpec) {
        const height = this.imgSize;  // 224
        const width = this.imgSize;   // 224
        const nMels = melSpec.length;
        const nFrames = melSpec[0].length;

        // Crear buffer para la imagen 224×224×3
        const imageData = new Float32Array(height * width * 3);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Mapear coordenadas de imagen a coordenadas del espectrograma
                // IMPORTANTE: En Python (cv2.resize), y=0 corresponde a m=0 (frecuencia baja).
                // No debemos invertir el eje Y para el tensor del modelo, de lo contrario verá el espectrograma de cabeza.
                const srcY = (y / (height - 1)) * (nMels - 1);
                const srcX = (x / (width - 1)) * (nFrames - 1);

                // Interpolación bilineal
                const x0 = Math.floor(srcX);
                const x1 = Math.min(x0 + 1, nFrames - 1);
                const y0 = Math.floor(srcY);
                const y1 = Math.min(y0 + 1, nMels - 1);
                const xFrac = srcX - x0;
                const yFrac = srcY - y0;

                const val =
                    melSpec[y0][x0] * (1 - xFrac) * (1 - yFrac) +
                    melSpec[y0][x1] * xFrac * (1 - yFrac) +
                    melSpec[y1][x0] * (1 - xFrac) * yFrac +
                    melSpec[y1][x1] * xFrac * yFrac;

                // Los 3 canales tienen el mismo valor (escala de grises para el modelo)
                const pixelIdx = (y * width + x) * 3;
                imageData[pixelIdx] = val;
                imageData[pixelIdx + 1] = val;
                imageData[pixelIdx + 2] = val;
            }
        }

        // Crear tensor 4D [batch=1, height=224, width=224, channels=3]
        return tf.tensor4d(imageData, [1, height, width, 3]);
    }

    /**
     * Extraer una ventana de 3 segundos del espectrograma y convertirla a tensor TF.js
     * @param {Float64Array[]} melSpec - Espectrograma completo
     * @param {number} startFrame - Frame de inicio
     * @param {number} nFramesSlice - Cantidad de frames a extraer
     * @returns {tf.Tensor4D} Tensor [1, 224, 224, 3] listo para inferencia
     */
    spectrogramSliceToTensor(melSpec, startFrame, nFramesSlice) {
        const height = this.imgSize;  // 224
        const width = this.imgSize;   // 224
        const nMels = melSpec.length;
        const nFramesTotal = melSpec[0].length;

        // Limitar nFramesSlice para no salirnos del rango
        const endFrame = Math.min(startFrame + nFramesSlice, nFramesTotal);
        const actualFramesSlice = endFrame - startFrame;

        const imageData = new Float32Array(height * width * 3);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcY = (y / (height - 1)) * (nMels - 1);
                const srcX = startFrame + (x / (width - 1)) * (actualFramesSlice - 1);

                const x0 = Math.floor(srcX);
                const x1 = Math.min(x0 + 1, nFramesTotal - 1);
                const y0 = Math.floor(srcY);
                const y1 = Math.min(y0 + 1, nMels - 1);
                const xFrac = srcX - x0;
                const yFrac = srcY - y0;

                const val =
                    melSpec[y0][x0] * (1 - xFrac) * (1 - yFrac) +
                    melSpec[y0][x1] * xFrac * (1 - yFrac) +
                    melSpec[y1][x0] * (1 - xFrac) * yFrac +
                    melSpec[y1][x1] * xFrac * yFrac;

                const pixelIdx = (y * width + x) * 3;
                imageData[pixelIdx] = val;
                imageData[pixelIdx + 1] = val;
                imageData[pixelIdx + 2] = val;
            }
        }

        return tf.tensor4d(imageData, [1, height, width, 3]);
    }

    // ════════════════════════════════════════════════════════════
    // VISUALIZACIÓN EN CANVAS
    // ════════════════════════════════════════════════════════════

    /**
     * Dibujar espectrograma completo en un canvas con colormap magma
     * @param {HTMLCanvasElement} canvas - Elemento canvas
     * @param {Float64Array[]} melSpec - Espectrograma [nMels × nFrames]
     */
    drawSpectrogram(canvas, melSpec) {
        const ctx = canvas.getContext('2d');
        const parentWidth = canvas.parentElement.clientWidth;
        const displayHeight = canvas.parentElement.clientHeight || 300;

        const nMels = melSpec.length;
        const nFrames = melSpec[0].length;

        // Calcular ancho dinámico: 3 segundos = ancho del padre. Más de 3s = más ancho
        const framesPer3s = (this.sampleRate * 3) / this.hopLength;
        const displayWidth = Math.max(parentWidth, parentWidth * (nFrames / framesPer3s));
        
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;

        // Ajustar resolución del canvas al tamaño en pantalla (para nitidez)
        const dpr = window.devicePixelRatio || 1;
        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;
        ctx.scale(dpr, dpr);

        // Márgenes para etiquetas de ejes
        const marginLeft = 45;
        const marginBottom = 25;
        const marginTop = 10;
        const marginRight = 10;

        const plotWidth = displayWidth - marginLeft - marginRight;
        const plotHeight = displayHeight - marginTop - marginBottom;

        // Fondo oscuro
        ctx.fillStyle = '#0a0e1a';
        ctx.fillRect(0, 0, displayWidth, displayHeight);

        // Dibujar espectrograma pixel por pixel
        const cellWidth = plotWidth / nFrames;
        const cellHeight = plotHeight / nMels;

        for (let m = 0; m < nMels; m++) {
            for (let t = 0; t < nFrames; t++) {
                const [r, g, b] = this._getColor(melSpec[m][t]);
                ctx.fillStyle = `rgb(${r},${g},${b})`;

                // Invertir eje Y: frecuencia baja abajo, alta arriba
                const x = marginLeft + t * cellWidth;
                const y = marginTop + (nMels - 1 - m) * cellHeight;

                ctx.fillRect(x, y, Math.ceil(cellWidth) + 1, Math.ceil(cellHeight) + 1);
            }
        }

        // Etiquetas de ejes
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';

        // Eje X: tiempo
        const duration = nFrames * this.hopLength / this.sampleRate;
        const numXTicks = 5;
        for (let i = 0; i <= numXTicks; i++) {
            const t = (duration * i / numXTicks).toFixed(1);
            const x = marginLeft + (i / numXTicks) * plotWidth;
            ctx.fillText(`${t}s`, x, displayHeight - 4);
        }

        // Eje Y: frecuencia
        ctx.textAlign = 'right';
        const freqs = [0, 1000, 2000, 3000, 4000];
        for (const freq of freqs) {
            const mel = this._hzToMel(freq);
            const melMin = this._hzToMel(this.fmin);
            const melMax = this._hzToMel(this.fmax);
            const ratio = (mel - melMin) / (melMax - melMin);
            if (ratio < 0 || ratio > 1) continue;
            const y = marginTop + (1 - ratio) * plotHeight;
            const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
            ctx.fillText(label, marginLeft - 5, y + 3);
        }

        // Título del eje Y
        ctx.save();
        ctx.translate(10, marginTop + plotHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillStyle = '#64748b';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText('Hz', 0, 0);
        ctx.restore();
    }

    // ════════════════════════════════════════════════════════════
    // VISUALIZACIÓN EN TIEMPO REAL (MICRÓFONO)
    // ════════════════════════════════════════════════════════════

    /**
     * Iniciar visualización en tiempo real del espectro desde el micrófono
     * Muestra un efecto de desplazamiento continuo (scrolling spectrogram)
     * @param {HTMLCanvasElement} canvas - Elemento canvas
     * @param {Function} getAudioFn - Función que retorna Uint8Array de datos de frecuencia
     */
    startRealtimeVisualization(canvas, getAudioFn) {
        this.stopRealtimeVisualization();

        const ctx = canvas.getContext('2d');
        const maxColumns = 200; // Número máximo de columnas visibles
        this._realtimeColumns = [];

        const draw = () => {
            this._realtimeAnimId = requestAnimationFrame(draw);

            const frequencyData = getAudioFn();
            if (!frequencyData) return;

            // Agregar nueva columna de datos (de 0 a nMels bins)
            const columnData = new Float64Array(this.nMels);
            const binRatio = frequencyData.length / this.nMels;
            for (let i = 0; i < this.nMels; i++) {
                const binIdx = Math.floor(i * binRatio);
                columnData[i] = frequencyData[binIdx] / 255.0;
            }
            this._realtimeColumns.push(columnData);

            // Limitar columnas al máximo
            if (this._realtimeColumns.length > maxColumns) {
                this._realtimeColumns.shift();
            }

            // Dibujar
            const displayWidth = canvas.clientWidth;
            const displayHeight = canvas.clientHeight;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = displayWidth * dpr;
            canvas.height = displayHeight * dpr;
            ctx.scale(dpr, dpr);

            // Fondo
            ctx.fillStyle = '#0a0e1a';
            ctx.fillRect(0, 0, displayWidth, displayHeight);

            const cols = this._realtimeColumns;
            if (cols.length === 0) return;

            const cellWidth = displayWidth / maxColumns;
            const cellHeight = displayHeight / this.nMels;

            for (let t = 0; t < cols.length; t++) {
                for (let m = 0; m < this.nMels; m++) {
                    const [r, g, b] = this._getColor(cols[t][m]);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;

                    const x = (maxColumns - cols.length + t) * cellWidth;
                    const y = (this.nMels - 1 - m) * cellHeight;

                    ctx.fillRect(x, y, Math.ceil(cellWidth) + 1, Math.ceil(cellHeight) + 1);
                }
            }
        };

        draw();
    }

    /**
     * Detener la visualización en tiempo real
     */
    stopRealtimeVisualization() {
        if (this._realtimeAnimId) {
            cancelAnimationFrame(this._realtimeAnimId);
            this._realtimeAnimId = null;
        }
        this._realtimeColumns = [];
    }
}
