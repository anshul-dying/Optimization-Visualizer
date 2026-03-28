import fs from "fs"
import path from "path"
import matter from "gray-matter"
import ReactMarkdown from "react-markdown"
import { notFound } from "next/navigation"
import { cn } from "@/lib/utils"
import { Mermaid } from "@/components/mermaid"
import { TOC } from "@/components/toc"

interface PageProps {
  params: {
    slug?: string[]
  }
}

interface TOCItem {
  id: string
  title: string
  level: number
}

function extractHeadings(content: string): TOCItem[] {
  const lines = content.split("\n")
  const headings: TOCItem[] = []
  
  lines.forEach(line => {
    const match = line.match(/^(#{1,3})\s+(.+)$/)
    if (match) {
      const level = match[1].length
      const title = match[2].trim()
      const id = title.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
      headings.push({ id, title, level })
    }
  })
  
  return headings
}

export default async function DocsPage({ params }: PageProps) {
  const slug = params.slug?.join("/") || "introduction"
  const filePath = path.join(process.cwd(), "docs", `${slug}.md`)

  if (!fs.existsSync(filePath)) {
    notFound()
  }

  const rawFileContent = fs.readFileSync(filePath, "utf8")
  // Strip HTML comments like <!-- prettier-ignore -->
  const fileContent = rawFileContent.replace(/<!--[\s\S]*?-->/g, '')
  
  const { content, data } = matter(fileContent)
  const headings = extractHeadings(content)

  return (
    <div className="relative flex">
      <div className="flex-1 max-w-3xl">
        <article className="prose prose-neutral dark:prose-invert max-w-none">
          <div className="mb-8 border-b pb-8">
            <h1 className="text-4xl font-extrabold tracking-tight text-primary mb-2">
              {data.title || slug.split("/").pop()?.replace(/-/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
            </h1>
            {data.description && (
              <p className="text-xl text-muted-foreground">{data.description}</p>
            )}
          </div>
          <div className="markdown-content">
            <ReactMarkdown
              components={{
                h1: ({ className, children, ...props }: any) => {
                  const id = String(children).toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
                  return <h1 id={id} className={cn("mt-12 scroll-m-20 text-4xl font-bold tracking-tight mb-4", className)} {...props}>{children}</h1>
                },
                h2: ({ className, children, ...props }: any) => {
                  const id = String(children).toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
                  return <h2 id={id} className={cn("mt-12 scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0 mb-4 text-primary", className)} {...props}>{children}</h2>
                },
                h3: ({ className, children, ...props }: any) => {
                  const id = String(children).toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
                  return <h3 id={id} className={cn("mt-8 scroll-m-20 text-2xl font-semibold tracking-tight mb-4", className)} {...props}>{children}</h3>
                },
                p: ({ className, ...props }) => (
                  <p className={cn("leading-7 [&:not(:first-child)]:mt-6 mb-4 text-muted-foreground", className)} {...props} />
                ),
                ul: ({ className, ...props }) => (
                  <ul className={cn("my-6 ml-6 list-disc [&>li]:mt-2", className)} {...props} />
                ),
                li: ({ className, ...props }) => (
                  <li className={cn("mt-2 text-muted-foreground", className)} {...props} />
                ),
                code: ({ className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || "")
                  if (match && match[1] === "mermaid") {
                    return <Mermaid chart={String(children).replace(/\n$/, "")} />
                  }
                  return (
                    <code className={cn("relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold", className)} {...props}>
                      {children}
                    </code>
                  )
                },
                pre: ({ className, ...props }) => (
                  <pre className={cn("mb-4 mt-6 overflow-x-auto rounded-lg border bg-card p-4", className)} {...props} />
                ),
                blockquote: ({ className, ...props }) => (
                  <blockquote className={cn("mt-6 border-l-4 border-primary pl-6 italic text-muted-foreground", className)} {...props} />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        </article>
      </div>

      {/* Dynamic Table of Contents */}
      <aside className="sticky top-24 hidden w-64 h-fit p-4 xl:block overflow-y-auto shrink-0 ml-8 border-l border-border/50">
        <div className="space-y-4">
          <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-primary/50">
            On this page
          </h4>
          <TOC headings={headings} />
        </div>
      </aside>
    </div>
  )
}
