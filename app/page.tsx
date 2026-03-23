import { Header } from "@/components/header"
import { HeroSection } from "@/components/hero-section"
import { OptimizerGrid } from "@/components/optimizer-grid"
import { InteractivePlayground } from "@/components/interactive-playground"
import { AdvancedMetrics } from "@/components/advanced-metrics"
import { AnalysisSection } from "@/components/analysis-section"
import { Footer } from "@/components/footer"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <OptimizerGrid />
        <InteractivePlayground />
        <AdvancedMetrics />
        <AnalysisSection />
      </main>
      <Footer />
    </div>
  )
}
