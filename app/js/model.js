// ============================================================
// CardioSound AI — Gestor del Modelo TensorFlow.js
// Carga, inferencia y gestión del modelo MobileNetV2
// ============================================================

/**
 * Clase para gestionar el modelo de clasificación de sonidos cardíacos
 * Carga un modelo MobileNetV2 convertido a TensorFlow.js Layers format
 */
export class ModelManager {
    constructor() {
        // Ruta del modelo (relativa a la raíz de la app)
        this.modelPath = './model/model.json';

        // Clases y configuración (debería coincidir con el entrenamiento)
        this.classNames = ['artifact', 'murmur', 'normal'];

        // Nombres para mostrar en la UI (en español)
        this.displayNames = {
            artifact: 'Artefacto / Ruido',
            murmur: 'Soplo Cardíaco',
            normal: 'Normal'
        };

        // Colores asociados a cada clase
        this.classColors = {
            artifact: '#6b7280',
            murmur: '#f59e0b',
            normal: '#10b981'
        };

        // Iconos de emojis para cada clase
        this.classIcons = {
            artifact: '🌫️',
            murmur: '⚠️',
            normal: '✅'
        };

        // Referencia al modelo cargado
        this._model = null;
        this._loaded = false;
    }

    /**
     * Cargar el modelo desde la ruta configurada
     * Realiza un warm-up con un tensor dummy para optimizar la primera inferencia real
     * @returns {Promise<boolean>} true si se cargó exitosamente
     */
    async load() {
        try {
            console.log('[ModelManager] Cargando modelo desde:', this.modelPath);

            // Cargar modelo en formato GraphModel (bypasses Keras 3 bugs)
            this._model = await tf.loadGraphModel(this.modelPath);
            console.log('[ModelManager] Modelo cargado exitosamente');

            // Warm-up: ejecutar una predicción con tensor dummy para compilar el grafo
            console.log('[ModelManager] Realizando warm-up...');
            const dummyInput = tf.zeros([1, 224, 224, 3]);
            let warmupResult = this._model.predict(dummyInput);
            if (Array.isArray(warmupResult)) { warmupResult = warmupResult[0]; }

            // Limpiar tensores de warm-up
            tf.dispose([dummyInput, warmupResult]);

            this._loaded = true;
            console.log('[ModelManager] Modelo listo para inferencia');
            return true;

        } catch (error) {
            console.error('[ModelManager] Error al cargar modelo:', error);
            this._loaded = false;

            // Determinar tipo de error para mensaje útil
            if (error.message && error.message.includes('404')) {
                throw new Error('Modelo no encontrado. Asegúrate de que model/model.json existe.');
            } else if (error.message && error.message.includes('fetch')) {
                throw new Error('No se pudo descargar el modelo. Verifica tu conexión.');
            } else {
                throw new Error(`Error al cargar el modelo: ${error.message}`);
            }
        }
    }

    /**
     * Ejecutar inferencia sobre un tensor de espectrograma
     * @param {tf.Tensor4D} spectrogramTensor - Tensor [1, 224, 224, 3]
     * @returns {Promise<Array<{label: string, displayName: string, score: number, color: string, icon: string}>>}
     *   Predicciones ordenadas por score descendente
     */
    async predict(spectrogramTensor) {
        if (!this._loaded || !this._model) {
            throw new Error('El modelo no está cargado. Cárgalo primero con load().');
        }

        // Ejecutar predicción
        let outputTensor = this._model.predict(spectrogramTensor);
        
        // GraphModel a veces retorna arreglos en lugar de tensor único
        if (Array.isArray(outputTensor)) {
            outputTensor = outputTensor[0];
        }

        // Obtener probabilidades como arreglo JavaScript
        const probabilities = await outputTensor.data();

        // Limpiar tensor de salida
        tf.dispose(outputTensor);

        // Construir arreglo de resultados
        const results = this.classNames.map((name, idx) => ({
            label: name,
            displayName: this.displayNames[name],
            score: probabilities[idx],
            color: this.classColors[name],
            icon: this.classIcons[name]
        }));

        // Ordenar por score descendente
        results.sort((a, b) => b.score - a.score);

        return results;
    }

    /**
     * Ejecutar inferencia usando ventana deslizante (Sliding Window) para audios largos
     * @param {Float64Array[]} melSpec - Espectrograma completo
     * @param {object} spectrogramEngine - Instancia de SpectrogramEngine para recortar ventanas
     * @returns {Promise<Array<{label: string, displayName: string, score: number, color: string, icon: string}>>}
     */
    async predictLongAudio(melSpec, spectrogramEngine) {
        if (!this._loaded || !this._model) {
            throw new Error('El modelo no está cargado. Cárgalo primero con load().');
        }

        const nFramesTotal = melSpec[0].length;
        // Calcular cuántos frames hay en 3 segundos
        const framesPer3s = Math.floor((3 * spectrogramEngine.sampleRate) / spectrogramEngine.hopLength);
        
        let startFrame = 0;
        // Sin solapamiento (Sliding Window desactivado). Se evalúan bloques de 3s seguidos.
        const hopFrames = framesPer3s; 
        
        const accumulatedProbs = new Float32Array(this.classNames.length);
        let numWindows = 0;

        while (startFrame < nFramesTotal || numWindows === 0) { // Garantizar al menos 1 iteración
            // Extraer ventana y convertir a tensor
            const inputTensor = spectrogramEngine.spectrogramSliceToTensor(melSpec, startFrame, framesPer3s);
            
            // Predicción
            let outputTensor = this._model.predict(inputTensor);
            if (Array.isArray(outputTensor)) {
                outputTensor = outputTensor[0];
            }
            const probs = await outputTensor.data();
            
            // Acumular
            for (let i = 0; i < probs.length; i++) {
                accumulatedProbs[i] += probs[i];
            }
            numWindows++;
            
            // Limpiar
            tf.dispose([inputTensor, outputTensor]);
            
            startFrame += hopFrames;
            
            // Si la siguiente ventana queda con muy pocos frames extras (< 10%), salir
            if (startFrame + (framesPer3s * 0.1) >= nFramesTotal) break;
        }

        // Promediar probabilidades
        for (let i = 0; i < accumulatedProbs.length; i++) {
            accumulatedProbs[i] /= numWindows;
        }

        // Construir arreglo de resultados
        const results = this.classNames.map((name, idx) => ({
            label: name,
            displayName: this.displayNames[name],
            score: accumulatedProbs[idx],
            color: this.classColors[name],
            icon: this.classIcons[name]
        }));

        // Ordenar por score descendente
        results.sort((a, b) => b.score - a.score);

        return results;
    }

    /**
     * Verificar si el modelo está cargado y listo
     * @returns {boolean}
     */
    isLoaded() {
        return this._loaded && this._model !== null;
    }

    /**
     * Liberar recursos del modelo
     */
    dispose() {
        if (this._model) {
            this._model.dispose();
            this._model = null;
            this._loaded = false;
            console.log('[ModelManager] Modelo liberado');
        }
    }
}
