/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#6D2842',
          light: '#8B4A5E',
          dark: '#4A1A2C',
        },
        secondary: {
          DEFAULT: '#9C6B7C',
          light: '#C4A4AF',
          dark: '#7A4D5C',
        },
        accent: '#D4AF37',
        success: '#2E7D32',
        warning: '#F57C00',
        danger: '#C62828',
      },
      fontFamily: {
        sans: ['Source Sans 3', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        display: ['Crimson Text', 'Georgia', 'serif'],
      },
      boxShadow: {
        'clinical': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
      },
    },
  },
  plugins: [],
}

