const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5001',
      changeOrigin: true,
    })
  );

  app.use(
    '/image',
    createProxyMiddleware({
      target: 'http://9gram.ddns.net:5002',
      changeOrigin: true,
    })
  );
};
