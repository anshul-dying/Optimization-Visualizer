import { Header } from "@/components/header"
import { MLPlayground } from "@/components/ml-playground"
import { Footer } from "@/components/footer"

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
