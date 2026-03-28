import { Header } from "@/components/layout/header"
import { MLPlayground } from "@/components/playground/ml-playground"
import { Footer } from "@/components/layout/footer"

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <MLPlayground />
      </main>
      <Footer />
    </div>
  )
}
