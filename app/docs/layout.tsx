"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { 
  Book, 
  Settings, 
  Cpu, 
  Layout, 
  Server, 
  Users, 
  ChevronRight,
  Info,
  Rocket,
  Code
} from "lucide-react"

const sidebarItems = [
  { 
    title: "Getting Started", 
    items: [
      { title: "Introduction", href: "/docs/introduction", icon: Info },
      { title: "Installation", href: "/docs/getting-started", icon: Rocket },
      { title: "Architecture", href: "/docs/architecture", icon: Cpu },
    ]
  },
  { 
    title: "Technical Reference", 
    items: [
      { title: "Algorithms", href: "/docs/algorithms", icon: Code },
      { title: "Frontend UI", href: "/docs/frontend", icon: Layout },
      { title: "Backend API", href: "/docs/backend", icon: Server },
    ]
  },
  { 
    title: "Community", 
    items: [
      { title: "Contributing", href: "/docs/contributing", icon: Users },
    ]
  }
]

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <div className="flex min-h-screen bg-background pt-16">
      {/* Modern Sidebar */}
      <aside className="fixed top-16 left-0 z-30 hidden w-72 h-[calc(100vh-4rem)] border-r bg-card md:block overflow-y-auto custom-scrollbar">
        <div className="p-8 space-y-8">
          {sidebarItems.map((group) => (
            <div key={group.title} className="space-y-3">
              <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-primary/50 px-3">
                {group.title}
              </h4>
              <nav className="space-y-1">
                {group.items.map((item) => {
                  const isActive = pathname === item.href || (pathname === "/docs" && item.href === "/docs/introduction")
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                        "group flex items-center gap-3 px-3 py-2.5 text-[13px] font-medium rounded-lg transition-all duration-200",
                        isActive 
                          ? "bg-primary/10 text-primary" 
                          : "text-muted-foreground hover:bg-primary/5 hover:text-foreground"
                      )}
                    >
                      <item.icon className={cn(
                        "w-4 h-4 transition-transform group-hover:scale-110",
                        isActive ? "text-primary" : "text-muted-foreground"
                      )} />
                      {item.title}
                    </Link>
                  )
                })}
              </nav>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 md:ml-72 bg-background min-h-screen">
        <div className="max-w-5xl px-12 py-16 mx-auto">
          <div className="docs-container animate-in fade-in slide-in-from-bottom-4 duration-500">
            {children}
            
            {/* Footer Navigation within Docs */}
            <div className="mt-20 pt-10 border-t border-border/50 flex justify-between items-center text-sm text-muted-foreground">
              <p>© 2025 Optimizer-Lens. Open Source Educational Tool.</p>
              <div className="flex gap-6">
                <Link href="/" className="hover:text-primary transition-colors">Home</Link>
                <Link href="/playground" className="hover:text-primary transition-colors">Playground</Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
