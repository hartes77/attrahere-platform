import React from 'react'

interface AttrahereLogoProps {
  className?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showText?: boolean
}

const AttrahereLogo: React.FC<AttrahereLogoProps> = ({
  className = '',
  size = 'md',
  showText = true,
}) => {
  const sizeMap = {
    sm: { container: 32, icon: 24, text: 16 },
    md: { container: 48, icon: 36, text: 20 },
    lg: { container: 72, icon: 54, text: 32 },
    xl: { container: 120, icon: 90, text: 64 },
  }

  const dimensions = sizeMap[size]

  return (
    <div className={`flex items-center gap-4 ${className}`}>
      {/* Logo Container with 3D Relief Effect */}
      <div
        className="relative rounded-2xl bg-gradient-to-br from-slate-200 to-slate-300 shadow-lg"
        style={{
          width: dimensions.container,
          height: dimensions.container,
          boxShadow: `
            inset 2px 2px 8px rgba(255, 255, 255, 0.8),
            inset -2px -2px 8px rgba(148, 163, 184, 0.6),
            4px 4px 16px rgba(0, 0, 0, 0.2)
          `,
        }}
      >
        {/* Inner highlight */}
        <div className="absolute inset-1 rounded-xl bg-gradient-to-br from-white/40 to-transparent" />

        {/* Logo Icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg
            width={dimensions.icon}
            height={dimensions.icon}
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              {/* Gradients for 3D effect */}
              <linearGradient id="aGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0f172a" />
                <stop offset="50%" stopColor="#1e293b" />
                <stop offset="100%" stopColor="#334155" />
              </linearGradient>

              <linearGradient id="skyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0ea5e9" />
                <stop offset="100%" stopColor="#0284c7" />
              </linearGradient>

              <linearGradient id="emeraldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="100%" stopColor="#059669" />
              </linearGradient>

              {/* Shadow filters */}
              <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow
                  dx="1"
                  dy="2"
                  stdDeviation="1"
                  floodColor="#000000"
                  floodOpacity="0.3"
                />
              </filter>
            </defs>

            {/* Letter A with relief effect */}
            <g filter="url(#shadow)">
              {/* Main A letter shape */}
              <path
                d="M25 75 L40 35 L60 35 L75 75 M35 60 L65 60"
                stroke="url(#aGradient)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />

              {/* Sky blue node (redesigned) */}
              <circle
                cx="65"
                cy="25"
                r="6"
                fill="url(#skyGradient)"
                stroke="#0284c7"
                strokeWidth="1"
              />
              <circle cx="65" cy="25" r="3" fill="#38bdf8" opacity="0.8" />

              {/* Emerald node (redesigned) */}
              <circle
                cx="75"
                cy="45"
                r="5"
                fill="url(#emeraldGradient)"
                stroke="#059669"
                strokeWidth="1"
              />
              <circle cx="75" cy="45" r="2.5" fill="#34d399" opacity="0.8" />

              {/* Connection lines with gradient */}
              <line
                x1="60"
                y1="35"
                x2="65"
                y2="25"
                stroke="#475569"
                strokeWidth="1.5"
                opacity="0.7"
              />
              <line
                x1="65"
                y1="60"
                x2="75"
                y2="45"
                stroke="#475569"
                strokeWidth="1.5"
                opacity="0.7"
              />
            </g>
          </svg>
        </div>
      </div>

      {/* Text with same styling as the site */}
      {showText && (
        <span
          className="font-bold text-slate-100 tracking-wide"
          style={{
            fontSize: dimensions.text,
            textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)',
          }}
        >
          Attrahere
        </span>
      )}
    </div>
  )
}

export default AttrahereLogo
export { AttrahereLogo }
