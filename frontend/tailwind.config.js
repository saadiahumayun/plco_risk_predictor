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
          DEFAULT: '#0B5394',
          light: '#1976D2',
          dark: '#073763',
        },
        secondary: {
          DEFAULT: '#00897B',
          light: '#4DB6AC',
          dark: '#00695C',
        },
        accent: '#7B1FA2',
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

