// BlindGuard Service Worker：仅缓存应用壳（静态资源），API/视频流永远走网络
const CACHE = 'blindguard-shell-v1';
const SHELL = [
    '/',
    '/manifest.json',
    '/icon_192.png',
    '/icon_512.png',
];

self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
    self.skipWaiting();
});

self.addEventListener('activate', (e) => {
    e.waitUntil(caches.keys().then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ));
    self.clients.claim();
});

self.addEventListener('fetch', (e) => {
    const url = new URL(e.request.url);
    // API 与视频流永不缓存
    if (url.pathname.startsWith('/api/') || url.pathname === '/video_feed') return;
    if (e.request.method !== 'GET' || url.origin !== location.origin) return;
    e.respondWith(
        caches.match(e.request).then((hit) => hit || fetch(e.request))
    );
});
