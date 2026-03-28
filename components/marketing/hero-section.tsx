import { Button } from "@/components/ui/button"
import { ArrowRight, Zap } from "lucide-react"

export function HeroSection() {
  return (
    <section className="py-20 px-4 text-center relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10" />
      <div className="container mx-auto relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
            <Zap className="w-4 h-4" />
            Interactive ML Optimization Visualizer
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-balance mb-6">
            Visualize ML{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent gradient-animate">
              Optimizers
            </span>{" "}
            in Real-Time
          </h1>

          <p className="text-xl text-muted-foreground text-balance mb-8 max-w-2xl mx-auto leading-relaxed">
            Explore gradient descent, SGD, Adam, and advanced optimization algorithms through interactive visualizations
            with comprehensive algorithmic analysis and performance comparisons.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground px-8" asChild>
              <a href="#playground">
                Start Visualizing
                <ArrowRight className="w-5 h-5 ml-2" />
              </a>
            </Button>
            <Button variant="outline" size="lg" className="px-8 bg-transparent" asChild>
              <a href="#analysis">View Analysis</a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
