/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // ── Phase 1: Spec Section 6.1 Typography ─────────────────────────────
      fontFamily: {
        // Display serif for headlines (the "one sentence" per region)
        serif:   ['Fraunces', 'Georgia', 'serif'],
        // Humanist sans for UI chrome, labels, body — Indic-ready
        sans:    ['Mukta', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        // Tabular figures for all data numbers, dates, coordinates
        mono:    ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        // Legacy aliases — keep for backward compat with existing components
        display: ['Fraunces', 'Georgia', 'serif'],
      },

      // ── Phase 1: Spec Section 6.2 Color Palette ──────────────────────────
      colors: {
        // Canvas & Surface (dark ops-room defaults)
        canvas:   '#0B1220',
        surface:  '#121B2E',
        ink:      '#E9EDF5',
        'ink-muted': '#8792A8',
        accent:   '#4FB8E0',

        // Light mode equivalents
        'canvas-light':  '#FAFAF8',
        'surface-light': '#FFFFFF',
        'ink-light':     '#14171F',

        // IMD Confidence Bands — reusing the institutional hue family
        imd: {
          high:         '#4C8C6D',   // p < 0.10  — muted terrain green "Trust the model"
          moderate:     '#C99A3A',   // 0.10–0.25 — monsoon-dust amber
          low:          '#B8551A',   // 0.25–0.50 — iron-oxide rust
          critical:     '#B8203A',   // p ≥ 0.50  — IMD red-alert red
          unprecedented:'#7C5CBF',  // OOD — violet, animated static texture
        },

        // Semantic risk aliases (match ConfidenceBand union type keys)
        risk: {
          HIGH:         '#4C8C6D',
          MODERATE:     '#C99A3A',
          LOW:          '#B8551A',
          CRITICAL:     '#B8203A',
          UNPRECEDENTED:'#7C5CBF',
        },

        // Dark canvas neutrals
        synoptic: {
          950: '#070B14',
          900: '#0B1220',
          850: '#0F1829',
          800: '#121B2E',
          750: '#162238',
          700: '#1E2D42',
          600: '#2D3F57',
          500: '#3D536E',
          400: '#5E7A96',
          300: '#8792A8',
          200: '#B8C5D6',
          100: '#D9E1ED',
          50:  '#E9EDF5',
        },
      },

      // ── Phase 1: Spec Section 6.3 Spacing & Radius ───────────────────────
      // 4px base unit: 4·8·12·16·24·32·48·64
      spacing: {
        '4.5': '1.125rem',
        '13':  '3.25rem',
        '15':  '3.75rem',
        '18':  '4.5rem',
      },
      borderRadius: {
        // Single radius used system-wide (6px)
        DEFAULT: '6px',
        sm: '4px',
        md: '6px',
        lg: '8px',
        xl: '12px',
        full: '9999px',
      },

      // ── Type Scale (Section 6.1 rem scale, 16px base) ────────────────────
      fontSize: {
        // Data display
        'display': ['2.5rem',   { lineHeight: '1.15', fontWeight: '600' }],
        'data-lg':  ['2.25rem', { lineHeight: '1.1',  letterSpacing: '-0.02em' }],
        'data-sm':  ['0.875rem',{ lineHeight: '1.4',  letterSpacing: '0' }],
        // Heading scale
        'h1':      ['1.75rem',  { lineHeight: '1.2',  fontWeight: '600' }],
        'h2':      ['1.25rem',  { lineHeight: '1.3',  fontWeight: '600' }],
        // Body
        'body':    ['0.9375rem',{ lineHeight: '1.6' }],
        'caption': ['0.8125rem',{ lineHeight: '1.4',  fontWeight: '500' }],
        // Existing tiny scales (keep for backward compat)
        '4xs': ['0.5rem',    { lineHeight: '0.65rem' }],
        '3xs': ['0.5625rem', { lineHeight: '0.75rem' }],
        '2xs': ['0.625rem',  { lineHeight: '0.875rem' }],
        xs:    ['0.75rem',   { lineHeight: '1rem' }],
        sm:    ['0.8125rem', { lineHeight: '1.25rem' }],
        base:  ['0.9375rem', { lineHeight: '1.6rem' }],
        lg:    ['1rem',      { lineHeight: '1.5rem' }],
        xl:    ['1.125rem',  { lineHeight: '1.75rem' }],
        '2xl': ['1.25rem',   { lineHeight: '1.875rem' }],
      },

      // ── Animation ─────────────────────────────────────────────────────────
      animation: {
        'pulse-slow':    'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'pulse-violet':  'pulseViolet 2.5s ease-in-out infinite',
        'ghost-pin':     'ghostPin 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards',
        'slide-in-right':'slideInRight 0.3s cubic-bezier(0.22, 1, 0.36, 1) forwards',
        'fade-in':       'fadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'streamline':    'flowStream 1.8s linear infinite',
      },
      keyframes: {
        pulseViolet: {
          '0%, 100%': { opacity: '0.7', transform: 'scale(1)' },
          '50%':      { opacity: '1',   transform: 'scale(1.02)' },
        },
        ghostPin: {
          '0%':   { transform: 'translateX(0)', opacity: '0' },
          '100%': { transform: 'translateX(var(--ghost-offset, 0))', opacity: '1' },
        },
        slideInRight: {
          '0%':   { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)',    opacity: '1' },
        },
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        flowStream: {
          to: { strokeDashoffset: '-32' },
        },
      },
    },
  },
  plugins: [],
};
