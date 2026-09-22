// BlindGuard Service Worker：仅缓存公开静态资源。
// 首页受鉴权保护且可能含令牌，API/视频流也永远走网络，绝不写入 Cache Storage。
const CACHE = 'blindguard-shell-v2';
const SHELL = [
    '/manifest.json',
    '/icon_192.png',
    '/icon_512.png',
];

self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE).then((cache) =>
        Promise.all(SHELL.map(async (path) => {
            const response = await fetch(path, { cache: 'no-cache' });
            if (response.ok) await cache.put(path, response);
        }))
    ));
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
    const protectedPath = url.pathname === '/' ||
        url.pathname === '/video_feed' ||
        url.pathname.startsWith('/api/');
    // 受保护资源、带令牌 URL、非 GET、跨域请求全部交给网络层。
    if (protectedPath || url.searchParams.has('token')) return;
    if (e.request.method !== 'GET' || url.origin !== location.origin) return;
    e.respondWith(caches.match(url.pathname).then((hit) => hit || fetch(e.request)));
});
