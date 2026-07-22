// ============================================================
// CardioSound AI — Service Worker para soporte offline (PWA)
// Estrategia: Cache-first local, Network-first CDN
// ============================================================

const CACHE_NAME = 'cardiosound-ai-v4';

// Recursos locales a pre-cachear
const LOCAL_ASSETS = [
    './',
    './index.html',
    './css/styles.css',
    './js/app.js',
    './js/audio.js',
    './js/spectrogram.js',
    './js/model.js',
    './js/ui.js',
    './manifest.json',
    './samples/normal_sample.wav',
    './samples/murmur_sample.wav',
    './samples/artifact_sample.wav'
];

// Recursos CDN
const CDN_ASSETS = [
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js'
];

// Todos los recursos combinados
const ALL_ASSETS = [...LOCAL_ASSETS, ...CDN_ASSETS];

// ── Evento Install: pre-cachear todos los recursos ──
self.addEventListener('install', (event) => {
    console.log('[SW] Instalando Service Worker v1...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Pre-cacheando recursos...');
                // Cachear cada recurso individualmente para no fallar si alguno no existe
                return Promise.allSettled(
                    ALL_ASSETS.map((url) =>
                        cache.add(url).catch((err) => {
                            console.warn(`[SW] No se pudo cachear: ${url}`, err.message);
                        })
                    )
                );
            })
            .then(() => {
                console.log('[SW] Instalación completada');
                return self.skipWaiting();
            })
    );
});

// ── Evento Activate: limpiar cachés antiguos ──
self.addEventListener('activate', (event) => {
    console.log('[SW] Activando Service Worker...');
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => {
                            console.log(`[SW] Eliminando caché antiguo: ${name}`);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => self.clients.claim())
    );
});

// ── Evento Fetch: estrategia de cacheo ──
self.addEventListener('fetch', (event) => {
    const requestUrl = new URL(event.request.url);

    // Para recursos CDN: Network-first, fallback a caché
    if (requestUrl.origin !== self.location.origin) {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Guardar copia en caché
                    const cloned = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, cloned);
                    });
                    return response;
                })
                .catch(() => {
                    // Sin red, intentar servir desde caché
                    return caches.match(event.request);
                })
        );
        return;
    }

    // Para recursos locales: Cache-first, fallback a red
    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    return cachedResponse;
                }
                return fetch(event.request).then((response) => {
                    // Cachear la nueva respuesta para futuras solicitudes
                    if (response.status === 200) {
                        const cloned = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, cloned);
                        });
                    }
                    return response;
                });
            })
            .catch(() => {
                // Si falla todo, devolver página principal como fallback
                if (event.request.destination === 'document') {
                    return caches.match('./index.html');
                }
            })
    );
});
