# CardioSound AI 🫀

CardioSound AI es una aplicación de inteligencia artificial capaz de clasificar y analizar sonidos cardíacos en tiempo real o a partir de archivos grabados (por ejemplo, con un estetoscopio electrónico). Utiliza un modelo **MobileNetV2** entrenado mediante *Transfer Learning* y ejecutado directamente en el dispositivo (Edge AI) gracias a **TensorFlow.js**.

La aplicación cuenta con una versión Web (PWA) y una versión Nativa para Android que permite forzar la salida de audio por el altavoz incluso si hay un estetoscopio conectado en el puerto Jack o USB-C.

---

## 📂 Estructura del Proyecto

El código está organizado en tres módulos principales:

*   **`app/`**: Contiene la Aplicación Web Progresiva (PWA). Está construida puramente con HTML, CSS y Javascript (Vanilla) sin empaquetadores complejos, lo que facilita su lectura. Adentro de esta carpeta se encuentra el modelo de IA.
*   **`mobile/`**: Contiene el envoltorio nativo para Android (construido con Capacitor JS). Aquí está el código fuente que interactúa con el sistema operativo.
*   **`training/`**: Contiene los scripts de Python para Google Colab que se utilizaron para entrenar y exportar el modelo original.
*   **Dataset**: El dataset de sonidos cardíacos utilizado para el entrenamiento (que incluye 3 datasets distintos más audios de YouTube) no se incluye en el repositorio por su tamaño. [Puedes descargarlo desde este enlace en Google Drive](https://drive.google.com/drive/folders/15Jy8kDC0d7Tc3JIIHvUGIyobwQhzj1Ko?usp=drive_link).
*   **`app.apk`**: Instalador APK para Android ya compilado. Puedes instalarlo directamente en tu celular sin necesidad de compilar el código fuente.
---

## 🚀 1. Ejecutar la Aplicación Web (PWA)

Como la aplicación utiliza **TensorFlow.js** y **Javascript Modules (ES6)**, no puedes simplemente abrir el archivo `index.html` con doble clic (por restricciones de seguridad del navegador CORS). Necesitas un servidor local.

### Usando VS Code (Recomendado)
1. Abre la carpeta raíz de este proyecto en Visual Studio Code.
2. Ve a la pestaña de extensiones e instala **Live Server**.
3. Explora los archivos, haz clic derecho sobre el archivo `app/index.html` y selecciona **"Open with Live Server"**.
4. La aplicación se abrirá en tu navegador predeterminado (usualmente en `http://localhost:5500`).

### Usando Python
Si tienes Python instalado, abre una terminal en la carpeta `app/` y ejecuta:
```bash
python -m http.server 8000
```
Luego abre en tu navegador web: `http://localhost:8000`

---

## 📱 2. Compilar e Instalar la Aplicación Android (APK)

La aplicación móvil es un "puente" nativo que envuelve la página web y le da superpoderes (acceso nativo al altavoz para puentear el estetoscopio y opciones de compartir nativas).

**Requisitos:** 
- Node.js y npm instalados.
- Android Studio instalado (con JDK configurado, preferiblemente JDK 21 o el Embedded JDK de Android Studio).

### Sincronizar cambios web con Android
Si hiciste cambios en los archivos de la carpeta `app/` (como modificar el HTML o el Javascript), debes sincronizarlos con la carpeta de Android antes de compilar.
Abre una terminal en la carpeta `mobile/` y ejecuta:
```bash
npm install
npx cap sync android
```

### Compilar el APK
Puedes compilar la app directamente usando la consola (sin abrir Android Studio). Abre una terminal en la carpeta `mobile/android/` y ejecuta:

**En Windows (PowerShell/CMD):**
```bash
.\gradlew assembleDebug
```

**En Mac/Linux:**
```bash
./gradlew assembleDebug
```

Si todo sale bien, encontrarás el archivo instalable en:
`mobile/android/app/build/outputs/apk/debug/app-debug.apk`

Copia este archivo a tu celular Android y ejecútalo para instalarlo.

---

## 🧠 3. Entrenar el Modelo de IA (Opcional)

Si deseas reentrenar el modelo con nuevos sonidos de estetoscopio:
1. Sube tu dataset a Google Drive.
2. Abre el archivo `training/CardioSound_Training.py` (o súbelo como un Notebook `.ipynb` a Google Colab).
3. Al finalizar, el script te dará una carpeta `modelo_tfjs/` con archivos `.json` y `.bin`.
4. Remplaza los archivos de la carpeta `app/model/` con estos nuevos archivos generados.

---

## 🔧 Soporte Especial (Hack de Hardware)

Si usas un estetoscopio conectado al puerto de auriculares de tu celular (que bloquea la salida de audio natural del teléfono), la app de Android cuenta con un puente en Java (`CardioNativePlugin`) que **fuerza** al hardware a reproducir el sonido por el altavoz exterior. Simplemente pulsa el botón de escuchar en la aplicación mientras el estetoscopio está conectado.
