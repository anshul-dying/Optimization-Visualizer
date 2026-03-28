"use client"

import React, { useEffect, useRef } from "react"
import mermaid from "mermaid"

interface MermaidProps {
  chart: string
}

export function Mermaid({ chart }: MermaidProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      theme: "default",
      securityLevel: "loose",
    })
    
    if (ref.current) {
      mermaid.contentLoaded()
    }
  }, [chart])

  useEffect(() => {
    if (ref.current) {
        ref.current.removeAttribute("data-processed")
        mermaid.render(`mermaid-${Math.random().toString(36).substr(2, 9)}`, chart).then(({ svg }) => {
            if (ref.current) {
                ref.current.innerHTML = svg
            }
        })
    }
  }, [chart])

  return (
    <div className="mermaid flex justify-center my-8 bg-white p-4 rounded-lg border border-border shadow-sm overflow-x-auto" ref={ref}>
      {chart}
    </div>
  )
}
