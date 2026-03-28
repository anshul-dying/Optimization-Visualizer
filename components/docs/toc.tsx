"use client"

import React, { useEffect, useState, useRef } from "react"
import { cn } from "@/lib/utils"

interface TOCItem {
  id: string
  title: string
  level: number
}

interface TOCProps {
  headings: TOCItem[]
}

export function TOC({ headings }: TOCProps) {
  const [activeId, setActiveId] = useState<string>("")
  const observer = useRef<IntersectionObserver | null>(null)

  useEffect(() => {
    const handleObserver = (entries: IntersectionObserverEntry[]) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setActiveId(entry.target.id)
        }
      })
    }

    observer.current = new IntersectionObserver(handleObserver, {
      rootMargin: "-100px 0% -80% 0%",
      threshold: 1.0,
    })

    const elements = headings.map((h) => document.getElementById(h.id))
    elements.forEach((el) => {
      if (el) observer.current?.observe(el)
    })

    return () => observer.current?.disconnect()
  }, [headings])

  return (
    <nav className="flex flex-col space-y-3">
      {headings.map((heading) => (
        <a
          key={heading.id}
          href={`#${heading.id}`}
          onClick={(e) => {
            e.preventDefault()
            document.getElementById(heading.id)?.scrollIntoView({
              behavior: "smooth"
            })
            window.history.pushState(null, "", `#${heading.id}`)
          }}
          className={cn(
            "text-[13px] transition-all duration-200 border-l-2 pl-4 -ml-[2px]",
            activeId === heading.id
              ? "text-primary border-primary font-medium"
              : "text-muted-foreground border-transparent hover:text-foreground hover:border-border",
            heading.level === 1 ? "mt-2" : "pl-6"
          )}
        >
          {heading.title}
        </a>
      ))}
    </nav>
  )
}
