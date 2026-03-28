import { Button } from "@/components/ui/button"
import { Github, Play } from "lucide-react"
import { ModeToggle } from "@/components/layout/mode-toggle"
import Link from "next/link"

export function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">ML</span>
            </div>
            <span className="font-bold text-xl text-foreground">Modulus</span>
          </div>

          <nav className="hidden md:flex items-center gap-6">
            <a href="/" className="text-muted-foreground hover:text-foreground transition-colors">
              Home
            </a>
            <a href="/#optimizers" className="text-muted-foreground hover:text-foreground transition-colors">
              Optimizers
            </a>
            <a href="/#analysis" className="text-muted-foreground hover:text-foreground transition-colors">
              Analysis
            </a>
            <a href="/playground" className="text-muted-foreground hover:text-foreground transition-colors">
              ML Playground
            </a>
            <Link href="/docs" className="text-muted-foreground hover:text-foreground transition-colors">
              Documentation
            </Link>
          </nav>

          <div className="flex items-center gap-2">
            <ModeToggle />
            <Button variant="ghost" size="sm" className="hidden sm:flex">
              <Github className="w-4 h-4 mr-2" />
              GitHub
            </Button>
            <Button size="sm" className="bg-primary hover:bg-primary/90">
              <Play className="w-4 h-4 mr-2" />
              Get Started
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
