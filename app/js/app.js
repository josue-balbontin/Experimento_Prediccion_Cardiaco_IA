// ============================================================
// CardioSound AI — Módulo Principal (Orquestador)
// Coordina todos los módulos: audio, espectrograma, modelo, UI
// ============================================================

import { AudioManager } from './audio.js';
import { SpectrogramEngine } from './spectrogram.js';
import { ModelManager } from './model.js';
import { UIManager } from './ui.js';

// ============================================================
// CONFIGURACIÓN GLOBAL
// ============================================================
const CONFIG = {
    // Configuración de audio
    SAMPLE_RATE: 22050,
    DURATION: 3,
    N_SAMPLES: 66150,

    // Configuración de espectrograma
    N_FFT: 2048,
    HOP_LENGTH: 512,
    N_MELS: 128,
    FMIN: 20,
    FMAX: 4000,
    IMG_SIZE: 224,

    // Clases
    CLASS_NAMES: ['artifact', 'murmur', 'normal'],
    DISPLAY_NAMES: {
        artifact: 'Artefacto / Ruido',
        murmur: 'Soplo Cardíaco',
        normal: 'Normal'
    },
    CLASS_COLORS: {
        artifact: '#6b7280',   // Gris
        murmur: '#f59e0b',     // Naranja
        normal: '#10b981'      // Verde
    },
    NUM_CLASSES: 3,

    // Rutas
    MODEL_PATH: './model/model.json',
    SAMPLE_PATHS: {
        artifact: 'samples/artifact_sample.wav',
        murmur: 'samples/murmur_sample.wav',
        normal: 'samples/normal_sample.wav'
    }
};

// ============================================================
// HELPERS DE DESCARGA Y COMPARTIR NATIVO
// ============================================================

async function downloadOrShareBlob(blob, filename, mimeType, title, text) {
    if (window.CardioAndroid) {
        try {
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = async () => {
                const base64data = reader.result.split(',')[1];
                window.CardioAndroid.shareBase64File(filename, base64data, mimeType);
            };
        } catch (e) {
            console.error('[App] Error al compartir nativo:', e);
            fallbackDownload(blob, filename);
        }
    } else {
        fallbackDownload(blob, filename);
    }
}

function fallbackDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

// ── Estado global de la aplicación ──
const state = {
    isRecording: false,
    isProcessing: false,
    isMonitoring: false,
    recordingStartTime: null,
    recordingTimerInterval: null,
    currentTab: 'mic'
};

// ── Instancias de módulos ──
let audioManager;
let spectrogramEngine;
let modelManager;
let ui;

// ============================================================
// INICIALIZACIÓN
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('[App] Iniciando CardioSound AI...');

    // Inicializar módulos
    audioManager = new AudioManager(CONFIG);
    spectrogramEngine = new SpectrogramEngine(CONFIG);
    modelManager = new ModelManager();
    ui = new UIManager();

    // Cachear elementos DOM
    ui.init();

    // Configurar event listeners
    setupTabNavigation();
    setupMicrophoneTab();
    setupFileTab();
    setupSamplesTab();
    setupAudioEvents();
    setupDownloadEvent();
    setupAudioDownloadEvent();
    setupAudioPlaybackEvent();
    setupDeviceSelectors();

    // Cargar modelo
    await loadModel();
});

// ============================================================
// CARGA DEL MODELO
// ============================================================

/**
 * Cargar el modelo de TensorFlow.js con manejo de errores y feedback visual
 */
async function loadModel() {
    ui.showLoading('Inicializando TensorFlow.js...');
    ui.setModelStatus('loading', 'Cargando...');

    try {
        // Esperar a que TF.js esté disponible
        await waitForTensorFlow();

        ui.updateLoadingMessage('Cargando modelo de clasificación...');

        // Intentar cargar el modelo
        await modelManager.load();

        // Éxito
        ui.setModelStatus('ready', 'Modelo listo');
        ui.hideLoading();
        ui.showToast('Modelo cargado correctamente', 'success');

    } catch (error) {
        console.warn('[App] No se pudo cargar el modelo:', error.message);

        // Mostrar error pero permitir que la app siga funcionando parcialmente
        ui.setModelStatus('error', 'Sin modelo');
        ui.hideLoading();
        ui.showToast(
            'Modelo no disponible. Genera el modelo primero con el script de entrenamiento.',
            'warning',
            6000
        );
    }
}

/**
 * Esperar a que TensorFlow.js esté disponible globalmente
 * @returns {Promise<void>}
 */
function waitForTensorFlow() {
    return new Promise((resolve, reject) => {
        // Si tf ya está disponible, resolver inmediatamente
        if (typeof tf !== 'undefined') {
            resolve();
            return;
        }

        // Esperar con timeout
        let elapsed = 0;
        const interval = setInterval(() => {
            if (typeof tf !== 'undefined') {
                clearInterval(interval);
                resolve();
            }
            elapsed += 100;
            if (elapsed > 15000) {
                clearInterval(interval);
                reject(new Error('Timeout esperando TensorFlow.js'));
            }
        }, 100);
    });
}

// ============================================================
// NAVEGACIÓN POR TABS
// ============================================================

/**
 * Configurar la lógica de cambio entre tabs
 */
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            if (targetTab === state.currentTab) return;

            // Detener grabación si se cambia de tab mientras graba
            if (state.isRecording) {
                stopRecording();
            }

            // Actualizar botones
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Actualizar contenido
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            const targetContent = document.getElementById(`tab-${targetTab}`);
            if (targetContent) targetContent.classList.add('active');

            state.currentTab = targetTab;
        });
    });
}

// ============================================================
// TAB: MICRÓFONO
// ============================================================

/**
 * Configurar eventos del tab de micrófono
 */
function setupMicrophoneTab() {
    const btnRecord = document.getElementById('btn-record');
    const btnMonitor = document.getElementById('btn-monitor');

    if (btnRecord) {
        btnRecord.addEventListener('click', () => {
            if (state.isProcessing) return;

            if (state.isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
    }

    if (btnMonitor) {
        btnMonitor.addEventListener('click', async () => {
            if (state.isProcessing) return;

            if (state.isRecording) {
                stopRecording();
            }

            if (state.isMonitoring) {
                stopMonitoring();
            } else {
                await startMonitoring();
            }
        });
    }
}

/**
 * Iniciar grabación desde el micrófono
 */
async function startRecording() {
    if (state.isRecording || state.isProcessing) return;

    if (state.isMonitoring) {
        stopMonitoring();
    }

    try {
        ui.resetResults();
        ui.toggleSpectrogramPlaceholder(false);
        const postRecordingControls = document.getElementById('post-recording-controls');
        if (postRecordingControls) postRecordingControls.classList.add('hidden');

        const inputId = document.getElementById('audio-input-select')?.value;
        const outputId = document.getElementById('audio-output-select')?.value;

        // Iniciar captura de audio
        await audioManager.startMicrophone(false, inputId, outputId);

        state.isRecording = true;
        state.recordingStartTime = Date.now();
        ui.setRecordingState(true);

        // Iniciar visualización en tiempo real
        spectrogramEngine.startRealtimeVisualization(
            document.getElementById('spectrogram-canvas'),
            () => audioManager.getAnalyserData()
        );

        // Timer de grabación
        state.recordingTimerInterval = setInterval(() => {
            const elapsed = (Date.now() - state.recordingStartTime) / 1000;
            ui.updateRecordingTimer(elapsed);
        }, 100);

    } catch (error) {
        state.isRecording = false;
        ui.setRecordingState(false);
        ui.showToast(error.message || 'Error al acceder al micrófono', 'error');
    }
}

/**
 * Detener grabación manualmente (se detiene automáticamente al llenar el buffer)
 */
function stopRecording() {
    if (!state.isRecording) return;

    // Detener timer
    clearInterval(state.recordingTimerInterval);
    state.recordingTimerInterval = null;

    // Detener visualización en tiempo real
    spectrogramEngine.stopRealtimeVisualization();

    // Obtener buffer antes de detener el micrófono
    const audioBuffer = audioManager.getAudioBuffer();

    // Detener micrófono
    audioManager.stopMicrophone();

    state.isRecording = false;
    ui.setRecordingState(false);

    // Verificar que hay suficientes muestras (al menos 0.5s)
    if (audioManager.samplesCollected < CONFIG.SAMPLE_RATE * 0.5) {
        ui.showToast('Grabación muy corta. Graba al menos 0.5 segundos.', 'warning');
        return;
    }

    // Procesar el audio capturado
    processAudio(audioBuffer);

    const postRecordingControls = document.getElementById('post-recording-controls');
    if (postRecordingControls) {
        postRecordingControls.classList.remove('hidden');
        postRecordingControls.style.display = 'flex';
    }
}

/**
 * Iniciar modo escucha (monitor)
 */
async function startMonitoring() {
    try {
        ui.resetResults();
        ui.toggleSpectrogramPlaceholder(false);
        const inputId = document.getElementById('audio-input-select')?.value;
        const outputId = document.getElementById('audio-output-select')?.value;
        
        await audioManager.startMicrophone(true, inputId, outputId);
        state.isMonitoring = true;

        const btnMonitor = document.getElementById('btn-monitor');
        if (btnMonitor) {
            btnMonitor.classList.add('active');
            btnMonitor.innerHTML = '🎧 Escuchando...';
        }
        document.getElementById('mic-status-text').textContent = 'Modo escucha activo (No grabando)';

        spectrogramEngine.startRealtimeVisualization(
            document.getElementById('spectrogram-canvas'),
            () => audioManager.getAnalyserData()
        );
    } catch (error) {
        state.isMonitoring = false;
        ui.showToast(error.message || 'Error al iniciar monitor', 'error');
    }
}

/**
 * Detener modo escucha
 */
function stopMonitoring() {
    if (!state.isMonitoring) return;

    spectrogramEngine.stopRealtimeVisualization();
    audioManager.stopMicrophone();
    state.isMonitoring = false;

    const btnMonitor = document.getElementById('btn-monitor');
    if (btnMonitor) {
        btnMonitor.classList.remove('active');
        btnMonitor.innerHTML = '🎧 Escuchar';
    }
    document.getElementById('mic-status-text').textContent = 'Listo para grabar';
}

/**
 * Configurar eventos del AudioManager
 */
function setupAudioEvents() {
    // Cuando el buffer de 3 segundos está listo (grabación automática completa)
    audioManager.on('bufferReady', (audioBuffer) => {
        // Detener timer y visualización
        clearInterval(state.recordingTimerInterval);
        state.recordingTimerInterval = null;
        spectrogramEngine.stopRealtimeVisualization();

        // Detener micrófono
        audioManager.stopMicrophone();

        state.isRecording = false;
        ui.setRecordingState(false);
        ui.showToast('Grabación completada', 'success');

        // Procesar
        processAudio(audioBuffer);

        const postRecordingControls = document.getElementById('post-recording-controls');
        if (postRecordingControls) {
            postRecordingControls.classList.remove('hidden');
            postRecordingControls.style.display = 'flex';
        }
    });

    // Errores del AudioManager
    audioManager.on('error', (err) => {
        ui.showToast(err.message, 'error');
    });

    // Progreso de grabación
    audioManager.on('recordingProgress', (progress) => {
        const elapsed = progress * CONFIG.DURATION;
        ui.updateRecordingTimer(elapsed);
    });
}

// ============================================================
// TAB: ARCHIVO
// ============================================================

/**
 * Configurar eventos del tab de archivo (drag & drop + input)
 */
function setupFileTab() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnClearFile = document.getElementById('btn-clear-file');

    if (!dropZone || !fileInput) return;

    // Click en la zona de drop abre el selector de archivo
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Evento de archivo seleccionado
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Eventos de drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Botón para limpiar archivo
    if (btnClearFile) {
        btnClearFile.addEventListener('click', () => {
            ui.hideFileInfo();
            ui.resetResults();
            ui.toggleSpectrogramPlaceholder(true);
            fileInput.value = '';
        });
    }
}

/**
 * Procesar un archivo de audio cargado por el usuario
 * @param {File} file - Archivo de audio
 */
async function handleFile(file) {
    // Validar tipo de archivo
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm',
                        'audio/x-wav', 'audio/wave'];
    const validExtensions = ['.wav', '.mp3', '.ogg', '.webm'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExt)) {
        ui.showToast('Formato de archivo no soportado. Usa WAV, MP3, OGG o WebM.', 'error');
        return;
    }

    ui.showFileInfo(file.name);
    ui.showToast(`Cargando: ${file.name}`, 'info');

    try {
        const audioData = await audioManager.loadAudioFile(file);
        processAudio(audioData);
    } catch (error) {
        console.error('[App] Error al cargar archivo:', error);
        ui.showToast(`Error al procesar el archivo: ${error.message}`, 'error');
    }
}

// ============================================================
// TAB: MUESTRAS
// ============================================================

/**
 * Configurar eventos del tab de muestras pre-cargadas
 */
function setupSamplesTab() {
    // Botones de reproducción
    document.querySelectorAll('.btn-sample-play').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const src = btn.dataset.src;
            if (!src) return;

            try {
                btn.textContent = '⏸ Pausar';
                await audioManager.playSample(src);
                // Restaurar texto después de un tiempo razonable
                setTimeout(() => { btn.textContent = '▶ Reproducir'; }, 5000);
            } catch (error) {
                btn.textContent = '▶ Reproducir';
                ui.showToast('Error al reproducir la muestra', 'error');
            }
        });
    });

    // Botones de análisis
    document.querySelectorAll('.btn-sample-analyze').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const src = btn.dataset.src;
            if (!src) return;

            if (state.isProcessing) {
                ui.showToast('Ya hay un análisis en progreso', 'warning');
                return;
            }

            btn.disabled = true;
            btn.textContent = '⏳ Analizando...';

            try {
                const audioData = await audioManager.loadAudioUrl(src);
                await processAudio(audioData);
                ui.showToast('Análisis de muestra completado', 'success');
            } catch (error) {
                console.error('[App] Error al analizar muestra:', error);
                ui.showToast(`Error: ${error.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🔬 Analizar';
            }
        });
    });
}

// ============================================================
// PIPELINE DE PROCESAMIENTO
// ============================================================

/**
 * Pipeline completo: audio → espectrograma → modelo → resultados
 * @param {Float32Array} audioData - Buffer de audio (66150 muestras a 22050 Hz)
 */
async function processAudio(audioData) {
    if (state.isProcessing) return;
    state.isProcessing = true;

    const canvas = document.getElementById('spectrogram-canvas');

    try {
        // 1. Calcular espectrograma Mel
        ui.showToast('Calculando espectrograma...', 'info', 2000);
        ui.toggleSpectrogramPlaceholder(false);

        const melSpec = spectrogramEngine.computeMelSpectrogram(audioData);

        // 2. Visualizar espectrograma en canvas
        spectrogramEngine.drawSpectrogram(canvas, melSpec);

        // 3. Verificar si el modelo está disponible
        if (!modelManager.isLoaded()) {
            ui.showToast('Modelo no cargado. Solo se muestra el espectrograma.', 'warning', 4000);
            state.isProcessing = false;
            return;
        }

        // 4. Ejecutar inferencia usando ventana deslizante (soporta cualquier duración)
        ui.showToast('Calculando diagnóstico general...', 'info', 1000);
        const predictions = await modelManager.predictLongAudio(melSpec, spectrogramEngine);

        // 5. Mostrar resultados en la UI
        ui.updateResults(predictions);

        console.log('[App] Predicciones:', predictions);

    } catch (error) {
        console.error('[App] Error en pipeline de procesamiento:', error);
        ui.showToast(`Error al procesar audio: ${error.message}`, 'error');
    } finally {
        state.isProcessing = false;
    }
}

// ============================================================
// DESCARGA DE ESPECTROGRAMA
// ============================================================

/**
 * Configurar el botón para descargar la imagen del espectrograma
 */
function setupDownloadEvent() {
    const btnDownload = document.getElementById('btn-download-spec');
    if (!btnDownload) return;

    btnDownload.addEventListener('click', () => {
        const canvas = document.getElementById('spectrogram-canvas');
        if (!canvas) return;

        try {
            canvas.toBlob(async (blob) => {
                if (!blob) {
                    ui.showToast('Error al generar la imagen', 'error');
                    return;
                }
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const filename = `espectrograma_${timestamp}.png`;
                await downloadOrShareBlob(blob, filename, 'image/png', 'Espectrograma', 'Espectrograma generado por CardioSound AI');
                ui.showToast('Espectrograma guardado', 'success', 2000);
            }, 'image/png');
        } catch (error) {
            console.error('[App] Error al descargar imagen:', error);
            ui.showToast('Error al procesar la descarga', 'error');
        }
    });
}

// ============================================================
// DESCARGA DE AUDIO GRABADO
// ============================================================

/**
 * Configurar el botón para descargar el audio grabado por el micrófono
 */
function setupAudioDownloadEvent() {
    const btnDownloadAudio = document.getElementById('btn-download-audio');
    if (!btnDownloadAudio) return;

    btnDownloadAudio.addEventListener('click', async () => {
        try {
            const wavBlob = audioManager.exportWAV();
            if (!wavBlob) {
                ui.showToast('No hay audio grabado para descargar', 'warning');
                return;
            }
            
            const timestamp = new Date().getTime();
            const filename = `cardiosound_grabacion_${timestamp}.wav`;
            
            await downloadOrShareBlob(wavBlob, filename, 'audio/wav', 'Audio del Latido', 'Grabación de latido de CardioSound AI');
            
            ui.showToast('Audio listo', 'success', 2000);
        } catch (error) {
            console.error('[App] Error al descargar audio:', error);
            ui.showToast('Error al descargar el audio', 'error');
        }
    });
}

/**
 * Configurar el botón para reproducir el audio grabado
 */
function setupAudioPlaybackEvent() {
    const btnPlayAudio = document.getElementById('btn-play-audio');
    if (!btnPlayAudio) return;

    btnPlayAudio.addEventListener('click', async () => {
        if (state.isPlaying) {
            audioManager.stopPlayback();
            state.isPlaying = false;
            btnPlayAudio.innerHTML = '▶ Reproducir';
            return;
        }

        const outputId = document.getElementById('audio-output-select')?.value;

        const success = await audioManager.playRecordedAudio(() => {
            state.isPlaying = false;
            btnPlayAudio.innerHTML = '▶ Reproducir';
        }, outputId);

        if (success) {
            state.isPlaying = true;
            btnPlayAudio.innerHTML = '⏹ Detener';
        } else {
            ui.showToast('No hay audio grabado para reproducir', 'warning');
        }
    });
}

// ============================================================
// CONFIGURACIÓN DE DISPOSITIVOS DE AUDIO
// ============================================================

async function setupDeviceSelectors() {
    const inputSelect = document.getElementById('audio-input-select');
    const outputSelect = document.getElementById('audio-output-select');
    
    if (!inputSelect || !outputSelect) return;
    
    try {
        // Pedir permiso temporalmente solo para poder ver los nombres (labels) de los dispositivos
        // Si ya tiene permiso, simplemente pasará rápido.
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch(e) {
            console.warn("No se pudo obtener permiso previo para enumerar dispositivos:", e);
        }

        const updateDeviceList = async () => {
            let devices = await navigator.mediaDevices.enumerateDevices();
            
            const currentInput = inputSelect.value;
            const currentOutput = outputSelect.value;

            inputSelect.innerHTML = '';
            outputSelect.innerHTML = '';
            
            let inputCount = 1;
            let outputCount = 1;
            
            devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                
                if (device.kind === 'audioinput') {
                    option.text = device.label || `Micrófono ${inputCount++}`;
                    inputSelect.appendChild(option);
                } else if (device.kind === 'audiooutput') {
                    option.text = device.label || `Altavoz ${outputCount++}`;
                    outputSelect.appendChild(option);
                }
            });
            
            if (outputSelect.options.length === 0) {
                const defaultOption = document.createElement('option');
                defaultOption.value = 'default';
                defaultOption.text = 'Predeterminado del sistema';
                outputSelect.appendChild(defaultOption);
            }
            if (inputSelect.options.length === 0) {
                const defaultOption = document.createElement('option');
                defaultOption.value = 'default';
                defaultOption.text = 'Predeterminado del sistema';
                inputSelect.appendChild(defaultOption);
            }

            // Restaurar selección previa si aún existe
            if (currentInput) inputSelect.value = currentInput;
            if (currentOutput) outputSelect.value = currentOutput;
        };

        await updateDeviceList();
        
        // Detener el stream temporal
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // Actualizar si el usuario conecta un nuevo micrófono/altavoz
        navigator.mediaDevices.addEventListener('devicechange', updateDeviceList);
        
    } catch (error) {
        console.error('Error al configurar selectores de dispositivo:', error);
    }
}

// ============================================================
// LIMPIEZA AL CERRAR
// ============================================================

window.addEventListener('beforeunload', () => {
    if (audioManager) audioManager.dispose();
    if (modelManager) modelManager.dispose();
    if (spectrogramEngine) spectrogramEngine.stopRealtimeVisualization();
});
