# FiAlerts Frontend - shadcn/ui Integration Guide

## ✅ Project Setup Complete

Your project now supports:
- ✅ **TypeScript** - Full TypeScript support enabled
- ✅ **Tailwind CSS** - Configured with PostCSS
- ✅ **shadcn/ui structure** - Component library structure in place

## 📁 Project Structure

```
frontend/
├── src/
│   ├── lib/
│   │   └── utils.ts              # cn() utility for merging Tailwind classes
│   ├── components/
│   │   ├── ui/
│   │   │   └── bg-gradient.tsx   # Background gradient component
│   │   └── demo.tsx              # Demo implementation
│   ├── App.js                     # Main app (can migrate to .tsx)
│   ├── index.js                   # Entry point (can migrate to .tsx)
│   └── index.css                  # Global styles with Tailwind directives
├── tsconfig.json                  # TypeScript configuration
├── tailwind.config.js             # Tailwind CSS configuration
└── postcss.config.js              # PostCSS configuration
```

## 🎨 Background Gradient Component

### Component Location
[src/components/ui/bg-gradient.tsx](src/components/ui/bg-gradient.tsx)

### Why `/components/ui`?
The `/components/ui` folder is the standard for shadcn/ui components because:
- **Consistency**: All shadcn/ui components use this convention
- **Organization**: Separates UI primitives from business logic components
- **Path imports**: Works seamlessly with `@/components/ui/*` imports
- **Future-proof**: Easy to add more shadcn/ui components later

### Component Props

```tsx
{
  className?: string;           // Additional Tailwind classes
  gradientFrom?: string;        // Starting color (default: "#fff")
  gradientTo?: string;          // Ending color (default: "#63e")
  gradientSize?: string;        // Gradient size (default: "125% 125%")
  gradientPosition?: string;    // Position (default: "50% 10%")
  gradientStop?: string;        // Color stop (default: "40%")
}
```

## 🚀 Usage Examples

### Basic Usage

```tsx
import { Component } from "@/components/ui/bg-gradient";

function MyComponent() {
  return (
    <div className="relative min-h-screen">
      <Component />
      {/* Your content here */}
    </div>
  );
}
```

### Custom Colors

```tsx
<Component 
  gradientFrom="#000"
  gradientTo="#1e40af"
  gradientPosition="0% 0%"
/>
```

### With Additional Styling

```tsx
<Component 
  className="opacity-50"
  gradientFrom="#fbbf24"
  gradientTo="#dc2626"
/>
```

## 🔧 Configuration Files

### TypeScript (`tsconfig.json`)
- Configured with path aliases: `@/*` → `./src/*`
- React JSX support enabled
- Strict type checking enabled

### Tailwind CSS (`tailwind.config.js`)
- Content paths configured for all `.js`, `.jsx`, `.ts`, `.tsx` files
- Custom CSS variables for border radius
- Dark mode support enabled with class strategy

### PostCSS (`postcss.config.js`)
- Tailwind CSS plugin configured
- Autoprefixer enabled for browser compatibility

## 📦 Installed Packages

```json
{
  "dependencies": {
    "clsx": "^2.1.1",
    "tailwind-merge": "^3.5.0"
  },
  "devDependencies": {
    "@types/node": "^25.4.0",
    "@types/react": "^19.2.14",
    "@types/react-dom": "^19.2.3",
    "typescript": "^5.9.3",
    "tailwindcss": "^3.x",
    "postcss": "^8.x",
    "autoprefixer": "^10.x"
  }
}
```

## 🎯 Integration with FiAlerts App

You can use the gradient background in your main App component:

```tsx
// In App.js or convert to App.tsx
import { Component as GradientBg } from "@/components/ui/bg-gradient";

function App() {
  return (
    <div className="relative min-h-screen">
      <GradientBg 
        gradientFrom="#0a0a0f"
        gradientTo="#6366f1"
        gradientPosition="50% 0%"
      />
      
      {/* Your existing app content */}
      <div className="relative z-10">
        {/* API Key form, chat interface, etc. */}
      </div>
    </div>
  );
}
```

## 🔄 Migrating to TypeScript

To fully migrate your app to TypeScript:

1. **Rename files**:
   ```bash
   mv src/App.js src/App.tsx
   mv src/index.js src/index.tsx
   ```

2. **Update imports** in `public/index.html`:
   - No changes needed - React Scripts handles this automatically

3. **Add type annotations** gradually to your components

## 🎨 Customization Examples

### Purple to Pink Gradient
```tsx
<Component 
  gradientFrom="#a855f7"
  gradientTo="#ec4899"
  gradientSize="150% 150%"
  gradientPosition="25% 25%"
/>
```

### Blue Ocean Theme
```tsx
<Component 
  gradientFrom="#dbeafe"
  gradientTo="#1e3a8a"
  gradientPosition="50% 0%"
  gradientStop="30%"
/>
```

### Dark Mode Gradient
```tsx
<Component 
  className="dark:opacity-70"
  gradientFrom="#000000"
  gradientTo="#312e81"
/>
```

## 🛠️ Development Commands

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## 📚 Next Steps

1. **Add more shadcn/ui components**: Button, Card, Dialog, etc.
2. **Install lucide-react**: `npm install lucide-react --legacy-peer-deps`
3. **Create a custom theme**: Extend Tailwind config with your brand colors
4. **Add animations**: Use Tailwind's animation utilities

## 🐛 Troubleshooting

### Path Alias Issues
If `@/` imports don't work, ensure:
- `tsconfig.json` has the correct `baseUrl` and `paths`
- Restart your development server

### Tailwind Classes Not Working
- Ensure Tailwind directives are in `index.css`
- Check that file paths in `tailwind.config.js` are correct
- Clear cache: `rm -rf node_modules/.cache`

### TypeScript Errors
- Run `npm install --legacy-peer-deps` for dependency conflicts
- Check that all `.tsx` files have proper type definitions

## 📖 Resources

- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)
- [TypeScript React Handbook](https://www.typescriptlang.org/docs/handbook/react.html)
