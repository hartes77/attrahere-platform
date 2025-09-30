# ⚛️ Attrahere Frontend (Next.js)

Modern, production-ready frontend for the Attrahere ML Code Quality Platform built with Next.js 15, TypeScript, and Tailwind CSS.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

**Access**: http://localhost:3000 (login: admin/admin123)

## 🔧 Development Tools

### Code Quality

```bash
# Format code
npx prettier --write "src/**/*.{ts,tsx,js,jsx,json,css,md}"

# Lint and fix
npx eslint "src/**/*.{ts,tsx}" --fix

# Type checking
npx tsc --noEmit
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env.local

# Required for development
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_BETA_GUARD_BYPASS=true
```

## 🎨 Design System

### Color Palette

Our carefully crafted palette is designed for developer tools with accessibility in mind:

- **Background**: `#0F172A` (Slate 950) - Main dark background
- **Surface**: `#1E293B` (Slate 800) - Cards, sidebar
- **Primary**: `#0EA5E9` (Cyan 500) - Trust & technology
- **Secondary**: `#22C55E` (Green 500) - AI & growth
- **Warning**: `#F59E0B` (Amber 500) - RLHF feedback
- **Error**: `#EF4444` (Red 500) - Critical issues

### Typography

- **Sans**: Inter - Clean, modern UI font
- **Mono**: JetBrains Mono - Code displays and technical content

## 🚀 Tech Stack

### Core Technologies

- **Next.js 14** with App Router and Turbopack
- **TypeScript** for type safety
- **Tailwind CSS** with custom design tokens
- **Radix UI** for accessible components
- **Playwright** for E2E testing

### Key Features

- 🎯 **Production-Ready** - Full JWT authentication & error handling
- 💡 **Real ML Analysis** - Live code analysis with pattern detection
- 🔄 **RLHF Integration** - User feedback collection for AI improvement
- ⚡ **Performance Optimized** - Async task polling with progress indicators
- 🔧 **Developer Experience** - Hot reload, type checking, comprehensive testing
- ♿ **Accessibility** - WCAG compliant design system
- 📊 **Bundle Analysis** - Monitor and optimize bundle size

## 🏗️ Architecture

```
/src
├── app/                     # Next.js App Router
│   ├── globals.css         # Global styles + design tokens
│   ├── layout.tsx          # Root layout with providers
│   └── page.tsx            # Main application page
├── components/
│   ├── ui/                 # Base design system components
│   │   ├── button.tsx      # Button variants & states
│   │   ├── card.tsx        # Card components with borders
│   │   ├── input.tsx       # Form input components
│   │   └── toast.tsx       # Notification system
│   └── features/           # Domain-specific components
│       ├── CodeAnalyzer.tsx   # Main analysis interface
│       ├── PatternCard.tsx    # ML pattern visualization
│       └── LoginForm.tsx      # JWT authentication
├── lib/
│   ├── api.ts              # Backend API client with auth
│   └── utils.ts            # Utilities + helper functions
├── types/
│   └── ml-analysis.ts      # Complete TypeScript schema
└── tests/
    └── e2e/                # Playwright E2E test suite
```

## 🛠️ Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Getting Started

```bash
# Install dependencies
npm install

# Start development server (with Turbopack)
npm run dev

# Type checking
npm run type-check

# Bundle analysis
npm run analyze

# Clean build artifacts
npm run clean
```

### Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Production build
- `npm run start` - Start production server
- `npm run lint` - ESLint checking
- `npm run analyze` - Bundle size analysis
- `npm run type-check` - TypeScript type checking
- `npm run test:e2e` - Run Playwright E2E tests
- `npm run test:e2e:ui` - Run E2E tests with UI

## 🎯 Performance Targets

### Core Web Vitals

- **LCP** < 2.5s (Largest Contentful Paint)
- **FID** < 100ms (First Input Delay)
- **CLS** < 0.1 (Cumulative Layout Shift)

### Bundle Size

- Base bundle < 500KB gzipped (without Monaco)
- Monaco editor lazy-loaded only when needed
- Aggressive code splitting for optimal caching

## 🧩 Component Library

### UI Components

All components follow the design system with consistent:

- Color variants (primary, secondary, warning, error)
- Size variants (xs, sm, default, lg)
- Accessibility features built-in
- TypeScript props for safety

### ML-Specific Components

- **CodeAnalyzer** - Main interface with code input & async analysis
- **PatternCard** - Rich visualization of detected ML anti-patterns
- **LoginForm** - JWT authentication with secure token storage
- **FeedbackWidget** - RLHF data collection integrated into pattern cards

### Production Features

- **JWT Authentication** - Secure login with admin/admin123 demo
- **Async Task Polling** - Real-time progress for long-running analysis
- **Error Boundaries** - Graceful error handling throughout the app
- **Loading States** - Professional UX with spinners and progress indicators
- **Timeout Handling** - 90-second timeout for Docker container setup

## 📡 Backend Integration

### API Proxy

Development proxy automatically forwards `/api/*` requests to backend:

```
Frontend (3000) → Backend (8002)
/api/v1/analyze → http://localhost:8002/api/v1/analyze
```

### Type Safety

TypeScript interfaces in `/types/ml-analysis.ts` match backend schemas:

- `MLPattern` - Pattern detection results
- `AnalysisResult` - Complete analysis response
- `UserFeedback` - RLHF feedback data

## 🔧 Configuration

### Tailwind Config

Custom tokens in `tailwind.config.ts`:

```typescript
colors: {
  background: "#0F172A",
  primary: "#0EA5E9",
  secondary: "#22C55E",
  // ... full palette
}
```

### Next.js Config

Optimizations in `next.config.js`:

- Bundle analyzer integration
- Code splitting for Monaco editor
- Security headers
- API proxy setup

## 🚀 Deployment

### Production Build

```bash
npm run build
npm run start
```

### Environment Variables

```env
# Optional: Backend URL override
NEXT_PUBLIC_API_URL=https://api.ml-platform.com
```

### Docker Support

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 📊 Monitoring

### Bundle Analysis

```bash
npm run analyze
# Opens bundle analyzer in browser
```

### Performance Monitoring

- Core Web Vitals tracking
- Bundle size monitoring
- Runtime performance metrics

## ✅ Current Status

### ✅ Phase 1: Foundation (COMPLETED)

- [x] Next.js 14 setup with App Router
- [x] Custom design system with Tailwind CSS
- [x] TypeScript integration with proper types
- [x] Radix UI component library

### ✅ Phase 2: Core Features (COMPLETED)

- [x] JWT authentication flow
- [x] Real ML code analysis integration
- [x] Async task polling with progress indicators
- [x] Pattern visualization with severity indicators
- [x] RLHF feedback collection system

### ✅ Phase 3: Production Ready (COMPLETED)

- [x] Comprehensive error handling
- [x] Professional loading states
- [x] E2E testing with Playwright
- [x] API client with proper timeout handling
- [x] Responsive design system

### 🚧 Next Phase: Advanced Features

- [ ] Fix suggestion application system
- [ ] Code Editor integration (Monaco)
- [ ] Real-time WebSocket updates
- [ ] Analytics dashboard
- [ ] Mobile optimizations

---

## 🔗 Links

- **Live Demo**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8002](http://localhost:8002)
- **Bundle Analyzer**: `npm run analyze`

---

**🚀 Status**: **Production-Ready** - Full-featured ML code analysis platform! ✅

**Demo**: Login with `admin/admin123` at [http://localhost:3000](http://localhost:3000)

**Latest Update**: Professional MLOps landing page deployed - September 21, 2025

Built with ❤️ for developer productivity and ML code quality.
