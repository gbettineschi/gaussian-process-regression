const CACHE = 'pyodide-v0275';

const PRECACHE = [
  'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.mjs',
  'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.asm.wasm',
  'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide_py.tar',
  'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/packages.json',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(PRECACHE)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  if (!event.request.url.includes('cdn.jsdelivr.net/pyodide')) return;

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(resp => {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(event.request, clone));
        return resp;
      });
    })
  );
});
