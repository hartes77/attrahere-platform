const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Temporarily disable ESLint during build for faster deployment
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Performance optimizations
  experimental: {
    // Enable turbopack for faster builds (already setup with create-next-app)
    turbo: {},
  },

  // Bundle optimization
  webpack: (config, { isServer }) => {
    // Optimize code splitting for Monaco editor and other heavy components
    if (!isServer) {
      config.optimization.splitChunks.cacheGroups = {
        ...config.optimization.splitChunks.cacheGroups,
        // Separate Monaco editor bundle
        monaco: {
          name: 'monaco',
          test: /[\\/]node_modules[\\/](@monaco-editor|monaco-editor)/,
          chunks: 'all',
          priority: 20,
        },
        // Separate vendor bundles for better caching
        vendor: {
          name: 'vendor',
          test: /[\\/]node_modules[\\/]/,
          chunks: 'all',
          priority: 10,
          maxSize: 244000, // ~240KB chunks
        },
      }
    }

    return config
  },

  // API proxy to backend during development
  async rewrites() {
    return [
      {
        source: '/api/health',
        destination: 'http://localhost:8001/health',
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8001/api/:path*',
      },
    ]
  },

  // Headers for security and performance
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ]
  },

  // Image optimization for any future assets
  images: {
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 3600,
  },
}

module.exports = withBundleAnalyzer(nextConfig)
