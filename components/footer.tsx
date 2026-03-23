import { Github, BookOpen, Mail } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-border bg-card/30 py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">ML</span>
              </div>
              <span className="font-bold text-xl">OptimizerViz</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Interactive visualization platform for understanding machine learning optimization algorithms.
            </p>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Algorithms</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Gradient Descent
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Stochastic GD
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Mini-Batch SGD
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Adam Optimizer
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Resources</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  API Reference
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Examples
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-foreground transition-colors">
                  Tutorials
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Connect</h3>
            <div className="flex gap-4">
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                <BookOpen className="w-5 h-5" />
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="border-t border-border mt-8 pt-8 text-center text-sm text-muted-foreground">
          <p>&copy; 2025 ML OptimizerViz. Built for educational purposes.</p>
        </div>
      </div>
    </footer>
  )
}
