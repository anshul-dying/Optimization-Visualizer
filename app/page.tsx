import { Header } from "@/components/layout/header"
import { HeroSection } from "@/components/marketing/hero-section"
import { OptimizerGrid } from "@/components/optimizers/optimizer-grid"
import { InteractivePlayground } from "@/components/playground/interactive"
import { AdvancedMetrics } from "@/components/analysis/advanced-metrics"
import { AnalysisSection } from "@/components/analysis/analysis-section"
import { Footer } from "@/components/layout/footer"

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
