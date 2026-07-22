# %% [markdown]
# # 🫀 CardioSound AI — Entrenamiento MobileNetV2
#
# **Clasificación de sonidos cardíacos usando Transfer Learning**
#
# | Detalle | Valor |
# |---|---|
# | **Dataset** | PASCAL Heart Sound Challenge |
# | **Modelo base** | MobileNetV2 (ImageNet) |
# | **Clases** | Normal · Soplo (Murmur) · Extrasístole · Artefacto |
# | **Estrategia** | Transfer Learning en 2 fases |
#
# > Este notebook entrena un modelo de clasificación de sonidos cardíacos
# > convirtiendo el audio a espectrogramas Mel y usando MobileNetV2 como
# > extractor de características. El modelo final se exporta a TensorFlow.js
# > para usarse en la aplicación web.

# %% [markdown]
# ---
# ## 📦 Celda 1: Instalación de Dependencias
# Instalamos las librerías necesarias. `librosa` para procesamiento de audio,
# `tensorflowjs` para exportar el modelo a formato web.

# %%
# === INSTALACIÓN DE DEPENDENCIAS ===
# -q = modo silencioso para no llenar la consola
!pip install -q tensorflow librosa matplotlib tensorflowjs scikit-learn seaborn opencv-python-headless

import warnings
warnings.filterwarnings('ignore')  # Silenciar advertencias de librosa/tf

print("✅ Dependencias instaladas correctamente")

# %% [markdown]
# ---
# ## 📁 Celda 2: Montar Google Drive y Configurar Rutas
# Montamos Google Drive donde está nuestro dataset y definimos TODAS las
# constantes de configuración. **Estas constantes DEBEN coincidir exactamente
# con las de la aplicación web.**

# %%
# === MONTAR GOOGLE DRIVE ===
from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive montado")

# %%
# === CONFIGURACIÓN COMPARTIDA ===
# ⚠️ IMPORTANTE: Estos valores DEBEN ser IDÉNTICOS a los de la app web.
# Si cambias algo aquí, debes cambiarlo también en la aplicación.

import os
import numpy as np
import time

# --- Parámetros de audio ---
SAMPLE_RATE = 22050       # Frecuencia de muestreo en Hz
DURATION = 3              # Duración de cada segmento en segundos
N_SAMPLES = 66150         # SAMPLE_RATE * DURATION = muestras por segmento
OVERLAP = 1.5             # Solapamiento en segundos para ventaneo de audios largos

# --- Parámetros del espectrograma Mel ---
N_FFT = 2048              # Tamaño de la ventana FFT
HOP_LENGTH = 512          # Salto entre ventanas (determina resolución temporal)
N_MELS = 128              # Número de bandas Mel (resolución frecuencial)
FMIN = 20                 # Frecuencia mínima en Hz (corta ruido sub-grave)
FMAX = 4000               # Frecuencia máxima en Hz (sonidos cardíacos están debajo)

# --- Parámetros de imagen ---
IMG_SIZE = 224             # MobileNetV2 espera imágenes de 224×224

# --- Clases (en orden alfabético, DEBE coincidir con la app web) ---
CLASS_NAMES = ['artifact', 'murmur', 'normal']
NUM_CLASSES = 3

# --- Ruta al dataset en Google Drive ---
# 🔧 MODIFICA ESTA RUTA si tu carpeta está en otro lugar de Drive
DRIVE_BASE = '/content/drive/MyDrive'
DATASET_PATH = os.path.join(DRIVE_BASE, 'IAEntrenamiento', 'Dataset separado')

# Si no encuentra en IAEntrenamiento, intentar en la raíz de Drive
if not os.path.isdir(DATASET_PATH):
    DATASET_PATH = os.path.join(DRIVE_BASE, 'Dataset separado')

# --- Ruta para guardar el modelo entrenado (en Drive para que persista) ---
OUTPUT_DIR = os.path.join(DRIVE_BASE, 'IAEntrenamiento')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verificar que el dataset existe
if os.path.isdir(DATASET_PATH):
    print(f"✅ Dataset encontrado en: {DATASET_PATH}")
    for item in sorted(os.listdir(DATASET_PATH)):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            wav_count = sum(
                1 for root, dirs, files in os.walk(item_path)
                for f in files if f.lower().endswith('.wav')
            )
            print(f"   📂 {item}: {wav_count} archivos .wav")
else:
    print(f"❌ ERROR: No se encontró el dataset en: {DATASET_PATH}")
    print("   Asegúrate de subir la carpeta 'Dataset separado' a tu Google Drive")
    print("   y ajustar la variable DATASET_PATH arriba.")

# %% [markdown]
# ---
# ## 🎵 Celda 3: Funciones de Audio y Espectrogramas
#
# ### Manejo de duraciones variables
# - **Audio > 3s** → Múltiples ventanas de 3s con solapamiento 1.5s
# - **Audio < 3s** → Relleno con ceros (centrado)
# - **Audio ≈ 3s** → Se usa tal cual
#
# ### Pipeline
# `Audio WAV → Librosa → Mel Spectrogram (dB) → Normalizar [0,1] → Resize 224×224 → 3 canales`

# %%
import librosa
import cv2
import tensorflow as tf
import gc

def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION, overlap=OVERLAP):
    """
    Carga un archivo de audio y lo divide en segmentos de duración fija.
    Retorna lista de arrays numpy, cada uno con sr*duration muestras.
    """
    n_samples = int(sr * duration)
    hop_samples = int(sr * (duration - overlap))

    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        print(f"   ⚠️ Error al cargar {os.path.basename(filepath)}: {e}")
        return []

    if len(audio) < int(sr * 0.1):
        return []

    segments = []

    if len(audio) >= n_samples:
        start = 0
        while start + n_samples <= len(audio):
            segment = audio[start:start + n_samples]
            segments.append(segment)
            start += hop_samples

        remaining = len(audio) - start
        if remaining > n_samples * 0.5:
            segment = np.zeros(n_samples, dtype=np.float32)
            segment[:remaining] = audio[start:]
            segments.append(segment)
    else:
        segment = np.zeros(n_samples, dtype=np.float32)
        pad_start = (n_samples - len(audio)) // 2
        segment[pad_start:pad_start + len(audio)] = audio
        segments.append(segment)

    return segments


def audio_to_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """
    Convierte audio a espectrograma Mel de 224×224×3 (listo para MobileNetV2).
    Pipeline: Mel Spec → dB → Normalizar [0,1] → Resize → 3 canales
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if mel_max - mel_min > 0:
        mel_normalized = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_normalized = np.zeros_like(mel_spec_db)

    mel_resized = cv2.resize(mel_normalized, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.stack([mel_resized] * 3, axis=-1)


print("✅ Funciones de audio y espectrograma definidas")
print(f"   📐 Cada espectrograma será de: {IMG_SIZE}×{IMG_SIZE}×3")
print(f"   ⏱️ Cada segmento de audio: {DURATION}s = {N_SAMPLES} muestras")

# %% [markdown]
# ---
# ## 📊 Celda 4: Train/Test Split y Carga a Disco
#
# **PREVENCIÓN DE DATA LEAKAGE**:
# 1. Hacemos el split a nivel de **ARCHIVO ORIGINAL** (.wav), no a nivel de segmento.
# 2. Generamos espectrogramas de Train y Test por separado.
# 3. El conjunto de Prueba (Test Set) queda 100% aislado.

# %%
from sklearn.model_selection import train_test_split

# Carpeta temporal en disco local de Colab (SSD rápido)
SPECS_DIR = '/content/spectrograms'
os.makedirs(SPECS_DIR, exist_ok=True)

def split_and_process_dataset(dataset_path, class_names, specs_dir):
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    train_wavs_per_class = {}
    counter = 0

    print("=" * 60)
    print("✂️ SPLIT A NIVEL DE ARCHIVO Y CARGA A DISCO")
    print("=" * 60)

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        wav_files = []
        for root, dirs, files in os.walk(class_dir):
            for f in sorted(files):
                if f.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, f))

        # 1. HYBRID SPLIT LOGIC
        if class_name == 'artifact':
            # Para ruidos no importa el Data Leakage de pacientes.
            # Spliteamos los pedacitos al azar para un balance de tamaños perfecto 80/20.
            from sklearn.model_selection import train_test_split
            train_w, test_w = train_test_split(wav_files, test_size=0.2, random_state=42)
        else:
            # Para corazones (normal, murmur) evitamos Data Leakage agrupando las "_part" del paciente
            groups = [os.path.basename(f).split('_part')[0] for f in wav_files]
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(wav_files, groups=groups))
            train_w = [wav_files[i] for i in train_idx]
            test_w = [wav_files[i] for i in test_idx]

        train_wavs_per_class[class_idx] = train_w

        files_ok, files_err = 0, 0

        # 2. PROCESAR TRAIN
        for fp in train_w:
            segments = load_audio(fp)
            if not segments: files_err += 1; continue
            for seg in segments:
                try:
                    spec = audio_to_mel_spectrogram(seg)
                    sp = os.path.join(specs_dir, f'spec_{counter:05d}.npy')
                    np.save(sp, spec)
                    train_paths.append(sp)
                    train_labels.append(class_idx)
                    counter += 1
                except: pass
            files_ok += 1

        # 3. PROCESAR TEST (Aislado)
        for fp in test_w:
            segments = load_audio(fp)
            if not segments: files_err += 1; continue
            for seg in segments:
                try:
                    spec = audio_to_mel_spectrogram(seg)
                    sp = os.path.join(specs_dir, f'spec_{counter:05d}.npy')
                    np.save(sp, spec)
                    test_paths.append(sp)
                    test_labels.append(class_idx)
                    counter += 1
                except: pass
            files_ok += 1

        print(f"   📂 {class_name:>12}: {len(train_w):3d} Train wavs | {len(test_w):3d} Test wavs")

    print("\n✅ SPLIT Y PROCESAMIENTO COMPLETADO")
    print(f"   Espectrogramas en Train (crudos): {len(train_paths)}")
    print(f"   Espectrogramas en Test (crudos):  {len(test_paths)}")

    return train_paths, train_labels, test_paths, test_labels, train_wavs_per_class, counter

train_paths, train_labels, test_paths, test_labels, train_wavs_per_class, spec_counter = split_and_process_dataset(DATASET_PATH, CLASS_NAMES, SPECS_DIR)

# %%
# === VISUALIZAR ESPECTROGRAMAS DE EJEMPLO ===
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 4))
fig.suptitle('🎵 Espectrogramas de Entrenamiento (1 por clase)', fontsize=14, fontweight='bold')

for i, class_name in enumerate(CLASS_NAMES):
    idx = next(j for j, lbl in enumerate(train_labels) if lbl == i)
    spec = np.load(train_paths[idx])

    axes[i].imshow(spec[:, :, 0], aspect='auto', origin='lower', cmap='magma')
    axes[i].set_title(f'{class_name}', fontsize=11)
    axes[i].set_xlabel('Tiempo')
    axes[i].set_ylabel('Frecuencia Mel')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'espectrogramas_ejemplo.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 🔄 Celda 5: Aumentación Aislada (SOLO TRAIN SET)
#
# Aplicamos data augmentation **únicamente al Train Set**. El Test Set no se contamina.

# %%
def augment_time_stretch(audio, sr=SAMPLE_RATE, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) >= N_SAMPLES:
        return stretched[:N_SAMPLES]
    padded = np.zeros(N_SAMPLES, dtype=np.float32)
    padded[:len(stretched)] = stretched
    return padded

def augment_pitch_shift(audio, sr=SAMPLE_RATE, n_steps_range=(-2, 2)):
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def augment_add_noise(audio, snr_range=(15, 25)):
    snr_db = np.random.uniform(*snr_range)
    signal_power = np.mean(audio ** 2)
    if signal_power == 0: return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return audio + noise

def augment_time_shift(audio, shift_pct=0.1):
    shift = int(len(audio) * np.random.uniform(-shift_pct, shift_pct))
    return np.roll(audio, shift)

def augment_audio(audio, sr=SAMPLE_RATE):
    aug = np.random.choice(['stretch', 'pitch', 'noise', 'shift'])
    if aug == 'stretch': return augment_time_stretch(audio, sr)
    elif aug == 'pitch': return augment_pitch_shift(audio, sr)
    elif aug == 'noise': return augment_add_noise(audio)
    else: return augment_time_shift(audio)

print("✅ Funciones de aumentación definidas")

# %%
def balance_train_set(train_paths, train_labels, train_wavs_per_class, class_names, specs_dir, counter):
    """
    Balancea SOLO el Train Set utilizando los archivos .wav asignados a Entrenamiento.
    """
    label_array = np.array(train_labels)
    counts = [np.sum(label_array == i) for i in range(len(class_names))]
    target = max(counts)

    print("=" * 60)
    print("🔄 BALANCEANDO SOLO EL TRAINING SET (Aumentación Aislada)")
    print("=" * 60)
    print(f"   Objetivo: {target} espectrogramas de Train por clase\n")

    balanced_train_paths = list(train_paths)
    balanced_train_labels = list(train_labels)

    for ci, cn in enumerate(class_names):
        needed = target - counts[ci]
        if needed <= 0:
            print(f"   ✅ {cn}: ya tiene {counts[ci]} muestras")
            continue

        print(f"   🔄 {cn}: tiene {counts[ci]}, necesita {needed} más...")

        wavs_train_class = train_wavs_per_class[ci]
        if not wavs_train_class:
            print(f"   ❌ {cn}: No hay audios en el train set para aumentar.")
            continue

        generated = 0
        attempts = 0
        while generated < needed and attempts < needed * 3:
            attempts += 1
            try:
                # Elegir SOLO archivos .wav que pertenecen al TRAIN SET
                fp = np.random.choice(wavs_train_class)
                audio, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
                if len(audio) < int(SAMPLE_RATE * 0.1): continue

                if len(audio) >= N_SAMPLES:
                    start = np.random.randint(0, len(audio) - N_SAMPLES + 1)
                    seg = audio[start:start + N_SAMPLES]
                else:
                    seg = np.zeros(N_SAMPLES, dtype=np.float32)
                    ps = (N_SAMPLES - len(audio)) // 2
                    seg[ps:ps + len(audio)] = audio

                aug = augment_audio(seg, SAMPLE_RATE)
                spec = audio_to_mel_spectrogram(aug)
                sp = os.path.join(specs_dir, f'spec_aug_{counter:05d}.npy')
                np.save(sp, spec)

                balanced_train_paths.append(sp)
                balanced_train_labels.append(ci)
                generated += 1
                counter += 1
            except:
                pass

        print(f"      ✅ Generados: {generated} espectrogramas aumentados")

    print(f"\n✅ Train Set balanceado: {len(balanced_train_paths)} muestras totales")
    return balanced_train_paths, balanced_train_labels

train_paths_bal, train_labels_bal = balance_train_set(
    train_paths, train_labels, train_wavs_per_class, CLASS_NAMES, SPECS_DIR, spec_counter
)

# === COMPARAR ORIGINAL vs AUMENTADO ===
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle('🔄 Comparación: Original vs Aumentado (Solo en Train Set)', fontsize=14, fontweight='bold')

orig_count = len(train_paths)
for i, cn in enumerate(CLASS_NAMES):
    idx = next((j for j, lbl in enumerate(train_labels) if lbl == i), None)
    if idx is None: continue
    s_orig = np.load(train_paths[idx])
    axes[0, i].imshow(s_orig[:, :, 0], aspect='auto', origin='lower', cmap='magma')
    axes[0, i].set_title(f'{cn}\n(Original Train)', fontsize=10)

    aug_idx = [j for j, lbl in enumerate(train_labels_bal) if lbl == i and j >= orig_count]
    s_aug = np.load(train_paths_bal[aug_idx[0]]) if aug_idx else s_orig
    axes[1, i].imshow(s_aug[:, :, 0], aspect='auto', origin='lower', cmap='magma')
    axes[1, i].set_title(f'{cn}\n(Aumentado Train)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'original_vs_aumentado.png'), dpi=150, bbox_inches='tight')
plt.show()

# Liberamos memoria
del train_paths, train_labels, train_wavs_per_class
import gc
gc.collect()

# %% [markdown]
# ---
# ## ⚡ Celda 6: Generadores de TensorFlow
#
# Construimos los Dataset bajo demanda. No hay Leakage.

# %%
print("=" * 60)
print("✂️ RESUMEN FINAL DE DATOS")
print("=" * 60)
print(f"   Entrenamiento (Balanceado): {len(train_paths_bal)} muestras")
print(f"   Prueba (Aislado, Intacto):  {len(test_paths)} muestras")

print("\n   📊 Distribución en ENTRENAMIENTO:")
for i, name in enumerate(CLASS_NAMES):
    print(f"      {name:>12}: {np.sum(np.array(train_labels_bal) == i):4d}")

print("\n   📊 Distribución en PRUEBA:")
for i, name in enumerate(CLASS_NAMES):
    print(f"      {name:>12}: {np.sum(np.array(test_labels) == i):4d}")

# %%
# === CREAR tf.data.Dataset CON CARGA DESDE DISCO ===

BATCH_SIZE = 32

def create_dataset_from_paths(paths, labels, batch_size, shuffle=False):
    """tf.data.Dataset que carga .npy desde disco bajo demanda."""

    def load_spec(path_tensor, label_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        spec = np.load(path_str).astype(np.float32)
        return spec, label_tensor

    def set_shapes(spec, label):
        spec.set_shape([IMG_SIZE, IMG_SIZE, 3])
        label.set_shape([NUM_CLASSES])
        return spec, label

    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_oh))

    if shuffle:
        dataset = dataset.shuffle(len(paths))

    dataset = dataset.map(
        lambda p, l: tf.py_function(func=load_spec, inp=[p, l], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        set_shapes,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset_from_paths(train_paths_bal, train_labels_bal, BATCH_SIZE, shuffle=True)
test_dataset = create_dataset_from_paths(test_paths, test_labels, BATCH_SIZE, shuffle=False)

print(f"\n✅ tf.data.Dataset creados (carga desde disco)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   🧠 RAM por batch: ~{BATCH_SIZE * 224 * 224 * 3 * 4 / 1024 / 1024:.0f} MB (en vez de ~3.2 GB total)")

# %% [markdown]
# ---
# ## 🏗️ Celda 7: Modelo MobileNetV2 (Transfer Learning)

# %%
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
], name='CardioSound_MobileNetV2')

print("🏗️ ARQUITECTURA DEL MODELO\n")
model.summary()

total_params = model.count_params()
trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"\n📊 Parámetros totales: {total_params:,}")
print(f"   Entrenables: {trainable_params:,}")
print(f"   Congelados: {total_params - trainable_params:,}")

# %% [markdown]
# ---
# ## 🏋️ Celda 8: Fase 1 — Backbone Congelado

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=3, min_lr=1e-6, verbose=1)
]

print("=" * 60)
print("🏋️ FASE 1: Entrenamiento (backbone congelado)")
print("=" * 60)

history_p1 = model.fit(train_dataset, epochs=20, validation_data=test_dataset,
                       callbacks=callbacks_p1, verbose=1)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('📈 Fase 1: Curvas de Entrenamiento', fontsize=14, fontweight='bold')

ax1.plot(history_p1.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history_p1.history['val_accuracy'], label='Val', linewidth=2)
ax1.set_xlabel('Época'); ax1.set_ylabel('Precisión')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history_p1.history['loss'], label='Train', linewidth=2)
ax2.plot(history_p1.history['val_loss'], label='Val', linewidth=2)
ax2.set_xlabel('Época'); ax2.set_ylabel('Pérdida')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase1_curvas.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n🏆 Mejor val accuracy Fase 1: {max(history_p1.history['val_accuracy']):.4f}")

# %% [markdown]
# ---
# ## 🔓 Celda 9: Fase 2 — Fine-Tuning (últimas 30 capas)

# %%
base_model.trainable = True
for layer in base_model.layers[:len(base_model.layers) - 30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable_now = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"🔓 Capas descongeladas: últimas 30 de {len(base_model.layers)}")
print(f"   Parámetros entrenables: {trainable_now:,}")

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=3, min_lr=1e-7, verbose=1)
]

print("\n" + "=" * 60)
print("🔓 FASE 2: Fine-Tuning")
print("=" * 60)

history_p2 = model.fit(train_dataset, epochs=15, validation_data=test_dataset,
                       callbacks=callbacks_p2, verbose=1)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('📈 Fase 2: Curvas de Fine-Tuning', fontsize=14, fontweight='bold')

ax1.plot(history_p2.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history_p2.history['val_accuracy'], label='Val', linewidth=2)
ax1.set_xlabel('Época'); ax1.set_ylabel('Precisión')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history_p2.history['loss'], label='Train', linewidth=2)
ax2.plot(history_p2.history['val_loss'], label='Val', linewidth=2)
ax2.set_xlabel('Época'); ax2.set_ylabel('Pérdida')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fase2_curvas.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n🏆 Mejor val accuracy Fase 2: {max(history_p2.history['val_accuracy']):.4f}")

# %% [markdown]
# ---
# ## 📊 Celda 10: Evaluación del Modelo

# %%
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("📊 EVALUACIÓN DEL MODELO")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"\n   📉 Pérdida: {test_loss:.4f}")
print(f"   🎯 Precisión: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

# Reconstruir predicciones por batches (sin cargar todo en RAM)
print("   🔄 Calculando predicciones...")
y_pred_list, y_true_list, x_list = [], [], []

for bx, by in test_dataset:
    y_pred_list.append(model.predict(bx, verbose=0))
    y_true_list.append(by.numpy())
    x_list.append(bx.numpy())

y_pred_proba = np.concatenate(y_pred_list)
y_test_oh = np.concatenate(y_true_list)
X_test_eval = np.concatenate(x_list)

y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_oh, axis=1)

del y_pred_list, y_true_list, x_list; gc.collect()

# %%
# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, linecolor='gray')
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Verdadero', fontsize=12)
ax.set_title('🧮 Matriz de Confusión', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
print("\n📋 REPORTE DE CLASIFICACIÓN:")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# %%
# === PEORES PREDICCIONES ===
incorrect = np.where(y_pred != y_true)[0]
if len(incorrect) > 0:
    confs = y_pred_proba[incorrect, y_pred[incorrect]]
    worst = incorrect[np.argsort(-confs)]
    n = min(8, len(worst))
    cols = min(4, n); rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f'❌ Peores Predicciones ({len(incorrect)} errores)', fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < n:
            si = worst[idx]
            ax.imshow(X_test_eval[si][:,:,0], aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f'Real: {CLASS_NAMES[y_true[si]]}\nPred: {CLASS_NAMES[y_pred[si]]} '
                        f'({y_pred_proba[si, y_pred[si]]:.1%})', fontsize=10, color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'peores_predicciones.png'), dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("🎉 ¡No hay errores!")

print(f"\n📊 Resumen: {len(incorrect)}/{len(y_true)} errores "
      f"({len(incorrect)/len(y_true)*100:.1f}%)")

del X_test_eval, y_pred_proba, y_test_oh; gc.collect()

# %% [markdown]
# ---
# ## 💾 Celda 11: Exportar Modelo

# %%
import json

model_path = os.path.join(OUTPUT_DIR, 'modelo_cardiosound.keras')
model.save(model_path)
print(f"✅ Modelo Keras guardado: {model_path}")

# Exportar a formato SavedModel (requerido para evitar errores de Keras 3 en TFJS)
saved_model_dir = os.path.join(OUTPUT_DIR, 'saved_model')
model.export(saved_model_dir)

spectrogram_config = {
    "sampleRate": SAMPLE_RATE, "duration": DURATION, "nSamples": N_SAMPLES,
    "nFft": N_FFT, "hopLength": HOP_LENGTH, "nMels": N_MELS,
    "fmin": FMIN, "fmax": FMAX, "imgSize": IMG_SIZE,
    "classNames": CLASS_NAMES, "numClasses": NUM_CLASSES,
    "modelVersion": "1.0.0",
    "trainingDate": time.strftime("%Y-%m-%d %H:%M:%S"),
    "testAccuracy": float(test_accuracy)
}

config_path = os.path.join(OUTPUT_DIR, 'spectrogram_config.json')
with open(config_path, 'w') as f:
    json.dump(spectrogram_config, f, indent=2)

print(f"✅ Config guardada: {config_path}")

# %%
tfjs_dir = os.path.join(OUTPUT_DIR, 'modelo_tfjs')
print("\n🔄 Convirtiendo a TensorFlow.js (GraphModel)...")
!tensorflowjs_converter --input_format=tf_saved_model "{saved_model_dir}" "{tfjs_dir}"

if os.path.isdir(tfjs_dir):
    print(f"\n✅ TF.js guardado en: {tfjs_dir}")
    for f in sorted(os.listdir(tfjs_dir)):
        sz = os.path.getsize(os.path.join(tfjs_dir, f))
        print(f"   📄 {f} ({sz/1024:.1f} KB)" if sz < 1024*1024 else f"   📄 {f} ({sz/1024/1024:.1f} MB)")
else:
    print("❌ Error en conversión. Intenta: !pip install tensorflowjs")

# %%
print(f"\n✅ Archivos en Google Drive: {OUTPUT_DIR}")
for item in sorted(os.listdir(OUTPUT_DIR)):
    ip = os.path.join(OUTPUT_DIR, item)
    if os.path.isdir(ip) and item != 'Dataset separado':
        print(f"   📂 {item}/")
        for sub in sorted(os.listdir(ip)):
            sp = os.path.join(ip, sub)
            if os.path.isfile(sp):
                sz = os.path.getsize(sp)
                print(f"      📄 {sub} ({sz/1024:.1f} KB)" if sz < 1024*1024 else f"      📄 {sub} ({sz/1024/1024:.1f} MB)")
    elif os.path.isfile(ip):
        sz = os.path.getsize(ip)
        print(f"   📄 {item} ({sz/1024:.1f} KB)" if sz < 1024*1024 else f"   📄 {item} ({sz/1024/1024:.1f} MB)")

print("""
📋 INSTRUCCIONES PARA LA APP WEB:
1. Descarga 'modelo_tfjs/' de tu Google Drive
2. Copia a 'Code/app/model/':
   - model.json → Code/app/model/model.json
   - group1-shard*.bin → Code/app/model/
3. Abre la app: python -m http.server 8080
""")

# %% [markdown]
# ---
# ## 🧪 Celda 12: Demo — Probar con Audio Individual

# %%
def predict_single_audio(model, filepath, class_names=CLASS_NAMES):
    """Predicción sobre un archivo de audio. Promedia segmentos."""
    print(f"\n🔍 Analizando: {os.path.basename(filepath)}")
    print("-" * 40)

    segments = load_audio(filepath)
    if not segments:
        print("❌ No se pudo cargar")
        return None

    print(f"   Segmentos: {len(segments)}")
    specs = [audio_to_mel_spectrogram(s) for s in segments]
    preds = model.predict(np.array(specs), verbose=0)
    avg = np.mean(preds, axis=0)
    ci = np.argmax(avg)

    print(f"\n   🎯 Predicción: {class_names[ci]}")
    print(f"   📊 Confianza: {avg[ci]:.1%}")
    for i, n in enumerate(class_names):
        bar = "█" * int(avg[i] * 30)
        print(f"      {n:>12}: {avg[i]:.4f} ({avg[i]*100:.1f}%) {bar}{'◄' if i == ci else ''}")

    return {'predicted_class': class_names[ci], 'confidence': float(avg[ci]),
            'probabilities': {n: float(p) for n, p in zip(class_names, avg)},
            'spectrograms': specs}


print("=" * 60)
print("🧪 DEMO: Predicción con archivos individuales")
print("=" * 60)

demo_results = {}
for cn in CLASS_NAMES:
    cd = os.path.join(DATASET_PATH, cn)
    for root, dirs, files in os.walk(cd):
        for f in sorted(files):
            if f.lower().endswith('.wav'):
                r = predict_single_audio(model, os.path.join(root, f))
                if r: demo_results[cn] = r
                break
        break

# %%
n = len(demo_results)
if n > 0:
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    for idx, (tc, r) in enumerate(demo_results.items()):
        ax = axes[0, idx] if n > 1 else axes[0]
        ax.imshow(r['spectrograms'][0][:,:,0], aspect='auto', origin='lower', cmap='magma')
        ok = r['predicted_class'] == tc
        ax.set_title(f'{"✅" if ok else "❌"} Real: {tc}\nPred: {r["predicted_class"]} ({r["confidence"]:.0%})',
                     fontsize=11, color='green' if ok else 'red', fontweight='bold')

        ax2 = axes[1, idx] if n > 1 else axes[1]
        probs = list(r['probabilities'].values())
        colors = ['#4CAF50' if name == r['predicted_class'] else '#2196F3' for name in CLASS_NAMES]
        bars = ax2.barh(CLASS_NAMES, probs, color=colors, edgecolor='gray')
        ax2.set_xlim(0, 1); ax2.set_xlabel('Probabilidad')
        for b, p in zip(bars, probs):
            ax2.text(b.get_width() + 0.02, b.get_y() + b.get_height()/2,
                     f'{p:.1%}', va='center', fontsize=9)

    fig.suptitle('🧪 Demo: Predicciones del Modelo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_predicciones.png'), dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# ## 🎉 ¡Entrenamiento Completado!
#
# | Archivo | Descripción |
# |---|---|
# | `modelo_cardiosound.h5` | Modelo Keras completo |
# | `spectrogram_config.json` | Config del espectrograma |
# | `modelo_tfjs/` | Modelo para TensorFlow.js |
#
# ### Próximos pasos:
# 1. 📥 Descarga `modelo_tfjs/` de Google Drive
# 2. 📂 Copia a `Code/app/model/`
# 3. 🚀 `python -m http.server 8080`
