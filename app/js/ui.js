// ============================================================
// CardioSound AI — Gestor de Interfaz de Usuario
// Manejo de UI: loading, resultados, toasts, estados
// ============================================================

/**
 * Clase para gestionar todas las interacciones de la interfaz de usuario
 * Controla overlay de carga, resultados, toasts, estados de grabación y más
 */
export class UIManager {
    constructor() {
        // Referencias a elementos del DOM (se cachean al inicializar)
        this._elements = {};
        this._toastCounter = 0;
    }

    /**
     * Inicializar el gestor de UI cacheando las referencias a los elementos del DOM
     */
    init() {
        this._elements = {
            // Overlay de carga
            loadingOverlay: document.getElementById('loading-overlay'),
            loadingMessage: document.getElementById('loading-message'),
            loadingBar: document.getElementById('loading-bar'),

            // Contenedor principal
            appContainer: document.getElementById('app-container'),

            // Estado del modelo
            modelStatus: document.getElementById('model-status'),
            modelStatusText: document.getElementById('model-status-text'),

            // Tab: Micrófono
            btnRecord: document.getElementById('btn-record'),
            recordingTimer: document.getElementById('recording-timer'),
            timerText: document.getElementById('timer-text'),
            micStatusText: document.getElementById('mic-status-text'),

            // Tab: Archivo
            fileInput: document.getElementById('file-input'),
            fileInfo: document.getElementById('file-info'),
            fileNameText: document.getElementById('file-name-text'),
            btnClearFile: document.getElementById('btn-clear-file'),
            dropZone: document.getElementById('drop-zone'),

            // Espectrograma
            spectrogramCanvas: document.getElementById('spectrogram-canvas'),
            spectrogramPlaceholder: document.getElementById('spectrogram-placeholder'),
            btnDownloadSpec: document.getElementById('btn-download-spec'),

            // Resultados
            resultsSection: document.getElementById('results-section'),
            diagnosisBanner: document.getElementById('diagnosis-banner'),
            diagnosisIcon: document.getElementById('diagnosis-icon'),
            diagnosisName: document.getElementById('diagnosis-name'),
            diagnosisConfidence: document.getElementById('diagnosis-confidence'),

            // Toasts
            toastContainer: document.getElementById('toast-container')
        };
    }

    // ════════════════════════════════════════════════════════════
    // OVERLAY DE CARGA
    // ════════════════════════════════════════════════════════════

    /**
     * Mostrar overlay de carga con un mensaje personalizado
     * @param {string} message - Mensaje a mostrar
     */
    showLoading(message = 'Cargando...') {
        const el = this._elements;
        if (el.loadingOverlay) {
            el.loadingOverlay.classList.remove('fade-out', 'hidden');
            el.loadingOverlay.style.display = 'flex';
        }
        if (el.loadingMessage) {
            el.loadingMessage.textContent = message;
        }
    }

    /**
     * Ocultar overlay de carga con animación de desvanecimiento
     */
    hideLoading() {
        const el = this._elements;
        if (el.loadingOverlay) {
            el.loadingOverlay.classList.add('fade-out');
            setTimeout(() => {
                el.loadingOverlay.style.display = 'none';
            }, 600);
        }
        if (el.appContainer) {
            el.appContainer.classList.remove('hidden');
        }
    }

    /**
     * Actualizar mensaje del overlay de carga
     * @param {string} message - Nuevo mensaje
     */
    updateLoadingMessage(message) {
        if (this._elements.loadingMessage) {
            this._elements.loadingMessage.textContent = message;
        }
    }

    // ════════════════════════════════════════════════════════════
    // ESTADO DEL MODELO
    // ════════════════════════════════════════════════════════════

    /**
     * Actualizar indicador de estado del modelo en el header
     * @param {'loading' | 'ready' | 'error'} status - Estado actual
     * @param {string} [message] - Mensaje descriptivo opcional
     */
    setModelStatus(status, message) {
        const el = this._elements;
        if (!el.modelStatus) return;

        // Eliminar todas las clases de estado previas
        el.modelStatus.classList.remove('status-loading', 'status-ready', 'status-error');

        switch (status) {
            case 'loading':
                el.modelStatus.classList.add('status-loading');
                el.modelStatusText.textContent = message || 'Cargando modelo...';
                break;
            case 'ready':
                el.modelStatus.classList.add('status-ready');
                el.modelStatusText.textContent = message || 'Modelo listo';
                break;
            case 'error':
                el.modelStatus.classList.add('status-error');
                el.modelStatusText.textContent = message || 'Error en modelo';
                break;
        }
    }

    // ════════════════════════════════════════════════════════════
    // ESTADO DE GRABACIÓN
    // ════════════════════════════════════════════════════════════

    /**
     * Cambiar la apariencia del botón de grabación y mostrar/ocultar timer
     * @param {boolean} isRecording - Si está grabando o no
     */
    setRecordingState(isRecording) {
        const el = this._elements;

        if (el.btnRecord) {
            el.btnRecord.classList.toggle('recording', isRecording);
        }

        if (el.recordingTimer) {
            el.recordingTimer.classList.toggle('hidden', !isRecording);
        }

        if (el.micStatusText) {
            el.micStatusText.textContent = isRecording
                ? 'Grabando sonido cardíaco...'
                : 'Listo para grabar';
        }
    }

    /**
     * Actualizar el texto del timer de grabación
     * @param {number} seconds - Segundos transcurridos
     */
    updateRecordingTimer(seconds) {
        if (this._elements.timerText) {
            this._elements.timerText.textContent = this.formatTime(seconds);
        }
    }

    /**
     * Formatear segundos a formato M:SS
     * @param {number} seconds - Segundos
     * @returns {string} Tiempo formateado
     */
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // ════════════════════════════════════════════════════════════
    // RESULTADOS DEL ANÁLISIS
    // ════════════════════════════════════════════════════════════

    /**
     * Mostrar resultados de la predicción con animaciones
     * @param {Array<{label: string, displayName: string, score: number, color: string, icon: string}>} predictions
     *   Arreglo de predicciones ordenadas por score descendente
     */
    updateResults(predictions) {
        const el = this._elements;

        // Mostrar sección de resultados
        if (el.resultsSection) {
            el.resultsSection.classList.remove('hidden');
        }

        // Actualizar cada tarjeta de resultado
        predictions.forEach((pred, idx) => {
            const nameEl = document.getElementById(`result-name-${idx}`);
            const scoreEl = document.getElementById(`result-score-${idx}`);
            const barEl = document.getElementById(`result-bar-${idx}`);
            const cardEl = document.getElementById(`result-card-${idx}`);

            if (nameEl) nameEl.textContent = `${pred.icon} ${pred.displayName}`;
            if (scoreEl) scoreEl.textContent = `${(pred.score * 100).toFixed(1)}%`;
            if (scoreEl) scoreEl.style.color = pred.color;

            if (barEl) {
                barEl.style.background = `linear-gradient(90deg, ${pred.color}, ${pred.color}aa)`;
                // Animar con un pequeño retraso para efecto cascada
                setTimeout(() => {
                    barEl.style.width = `${(pred.score * 100).toFixed(1)}%`;
                }, 100 + idx * 80);
            }

            // Stripe de color en la tarjeta
            if (cardEl) {
                const stripe = cardEl.querySelector('.result-card-stripe');
                if (stripe) stripe.style.background = pred.color;

                // Marcar la tarjeta líder con efecto glow
                cardEl.classList.toggle('leading', idx === 0);
            }
        });

        // Actualizar banner de diagnóstico
        if (predictions.length > 0) {
            const top = predictions[0];
            this._updateDiagnosisBanner(top);
        }
    }

    /**
     * Actualizar banner de diagnóstico principal con la predicción dominante
     * @param {object} prediction - Predicción principal
     * @private
     */
    _updateDiagnosisBanner(prediction) {
        const el = this._elements;

        if (el.diagnosisBanner) {
            el.diagnosisBanner.classList.remove('hidden');

            // Remover todas las clases de glow previas
            el.diagnosisBanner.classList.remove(
                'glow-normal', 'glow-murmur', 'glow-artifact'
            );
            // Agregar glow según la clase dominante
            el.diagnosisBanner.classList.add(`glow-${prediction.label}`);
        }

        if (el.diagnosisIcon) el.diagnosisIcon.textContent = prediction.icon;
        if (el.diagnosisName) {
            el.diagnosisName.textContent = prediction.displayName;
            el.diagnosisName.style.color = prediction.color;
        }
        if (el.diagnosisConfidence) {
            el.diagnosisConfidence.textContent = `${(prediction.score * 100).toFixed(1)}%`;
            el.diagnosisConfidence.style.color = prediction.color;
        }
    }

    /**
     * Resetear todos los paneles de resultados a su estado inicial
     */
    resetResults() {
        const el = this._elements;

        if (el.resultsSection) {
            el.resultsSection.classList.add('hidden');
        }

        if (el.diagnosisBanner) {
            el.diagnosisBanner.classList.add('hidden');
            el.diagnosisBanner.classList.remove(
                'glow-normal', 'glow-murmur', 'glow-artifact'
            );
        }

        // Resetear tarjetas
        for (let i = 0; i < 4; i++) {
            const nameEl = document.getElementById(`result-name-${i}`);
            const scoreEl = document.getElementById(`result-score-${i}`);
            const barEl = document.getElementById(`result-bar-${i}`);
            const cardEl = document.getElementById(`result-card-${i}`);

            if (nameEl) nameEl.textContent = '—';
            if (scoreEl) { scoreEl.textContent = '0%'; scoreEl.style.color = ''; }
            if (barEl) { barEl.style.width = '0%'; barEl.style.background = ''; }
            if (cardEl) cardEl.classList.remove('leading');
        }
    }

    // ════════════════════════════════════════════════════════════
    // ESPECTROGRAMA
    // ════════════════════════════════════════════════════════════

    /**
     * Mostrar u ocultar el placeholder del espectrograma
     * @param {boolean} show - Si se debe mostrar el placeholder
     */
    toggleSpectrogramPlaceholder(show) {
        if (this._elements.spectrogramPlaceholder) {
            this._elements.spectrogramPlaceholder.style.display = show ? 'flex' : 'none';
        }
        if (this._elements.btnDownloadSpec) {
            if (show) {
                this._elements.btnDownloadSpec.classList.add('hidden');
            } else {
                this._elements.btnDownloadSpec.classList.remove('hidden');
            }
        }
    }

    // ════════════════════════════════════════════════════════════
    // ARCHIVO DE AUDIO
    // ════════════════════════════════════════════════════════════

    /**
     * Mostrar información del archivo cargado
     * @param {string} fileName - Nombre del archivo
     */
    showFileInfo(fileName) {
        const el = this._elements;
        if (el.fileInfo) el.fileInfo.classList.remove('hidden');
        if (el.fileNameText) el.fileNameText.textContent = fileName;
    }

    /**
     * Ocultar información del archivo
     */
    hideFileInfo() {
        const el = this._elements;
        if (el.fileInfo) el.fileInfo.classList.add('hidden');
        if (el.fileNameText) el.fileNameText.textContent = '';
    }

    // ════════════════════════════════════════════════════════════
    // TOASTS (NOTIFICACIONES)
    // ════════════════════════════════════════════════════════════

    /**
     * Mostrar una notificación toast
     * @param {string} message - Mensaje a mostrar
     * @param {'success' | 'error' | 'info' | 'warning'} type - Tipo de notificación
     * @param {number} duration - Duración en ms (default: 4000)
     */
    showToast(message, type = 'info', duration = 4000) {
        const container = this._elements.toastContainer;
        if (!container) return;

        // Iconos según tipo
        const icons = {
            success: '✅',
            error: '❌',
            info: 'ℹ️',
            warning: '⚠️'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.id = `toast-${++this._toastCounter}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
            <span class="toast-message">${message}</span>
        `;

        container.appendChild(toast);

        // Auto-remove después de la duración
        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, duration);
    }
}
