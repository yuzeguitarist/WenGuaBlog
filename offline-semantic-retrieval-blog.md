# Why I didn't want my diary in the cloud

*Notes on offline semantic search + SwiftUI accessibility*

You can use the code in this post directly. If you publish something based on it, please leave attribution.

**English · 简体中文**  
Jump to: [English](#english) · [简体中文](#简体中文)

---

## 流程图

![On-device semantic retrieval pipeline](assets/pipeline.svg)

![Accessibility flow for VoiceOver and Reduce Motion](assets/a11y-flow.svg)

---

# English

## Why I didn't want my diary in the cloud

Before orchestra rehearsal, I can sit for twenty minutes stuck in this loop:

*If I tell them the tempo feels wrong, will it help, or will everyone think I'm being annoying?  
If I stay quiet, will I feel better, or worse?*

Overthinking isn't stupidity. It often means you have too many thoughts and nowhere to place them.

That's why I made Wen Gua. You type what's bothering you, draw a hexagram, and get a reading that helps you think through the mess. The whole flow takes about three minutes: long enough to help, short enough not to become another way to avoid deciding.

When I started building this, I realized two things had to be true:

1. **The words you type need to stay on your phone.**  
   If you're making an app to help people think through worries, uploading those worries to a server feels wrong.
2. **It needs to work for everyone.**  
   I learned this the hard way when I realized my first version was almost unusable for anyone using VoiceOver.

This post is basically my notes on the two hardest parts:

- **Making search work offline** using Apple's `NaturalLanguage` framework
- **Making the UI actually accessible** for VoiceOver and Reduce Motion

No servers, no accounts, no tracking. Just your phone doing the work.

---

## What "semantic search" actually is

"Semantic search" sounds fancy, but it means one simple thing:

> If two sentences mean the same thing, the app should know they're related—even if they use completely different words.

Normal search looks for exact words. Semantic search looks for meaning.

You do this with an **embedding**: turn text into numbers, and if two pieces of text have similar meanings, their vectors stay close together.

That sounds abstract at first. Think of it as coordinates on a map, except the map represents meanings instead of places.

For Wen Gua, I only needed to search through a few hundred short snippets I wrote, so I kept it simple:

- Keep the database small
- Write precise snippets
- Keep the search code simple
- Make everything work offline

---

## What NaturalLanguage and NLEmbedding do

Apple's **NaturalLanguage** framework provides on-device text tools: language detection, word tokenization, and part-of-speech tagging.

The specific part I needed is `NLEmbedding`, which turns text into those number coordinates I mentioned. There are two main ones:

- `NLEmbedding.sentenceEmbedding(for:)` for whole sentences
- `NLEmbedding.wordEmbedding(for:)` for individual words

One thing I learned the hard way: these embeddings are not always available. They might not exist for every language or every device, so your code has to handle that path.

---

## How I built it

Some people would call this "RAG" (retrieval-augmented generation). In Wen Gua, there is no large model writing content from scratch. I wrote all the readings myself, and the app retrieves the best matches.

I actually prefer it this way. It feels more honest.

Here's how it works:

1. **Parse** what the person typed to make it cleaner for searching
2. **Sense** the tone (so I can respond with the right feeling)
3. **Embed** the question into numbers
4. **Retrieve** the best matching readings
5. **Compose** everything into a structured response
6. **Format** it so it looks good and works with VoiceOver

---

## Step 1: Getting the embedding

```swift
import NaturalLanguage

func sentenceVector(for text: String) -> [Double]? {
    guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
        return nil
    }
    return embedding.vector(for: text)
}
```

Two things that took me forever to figure out:

- `NLEmbedding.sentenceEmbedding(for:)` can fail and return `nil`. That's normal.
- The result is `[Double]?` but I found converting to `Float` saves memory and makes the math faster.

---

## Step 2: Caching embeddings (this made everything way faster)

I was recomputing the same embeddings over and over. It was so slow.

Then I learned you can cache them:

- Put query embeddings in an `NSCache`
- Store both the numbers and the "length" (called L2 norm)
- Make sure it's thread-safe because Swift is picky about that

```swift
import Foundation
import NaturalLanguage

/// Thread-safe, shared embedding cache.
final class SentenceEmbeddingCache: @unchecked Sendable {
    static let shared = SentenceEmbeddingCache()

    struct Entry {
        let vector: [Float]
        let norm: Float
    }

    private let embedding = NLEmbedding.sentenceEmbedding(for: .english)
    private let cache = NSCache<NSString, EmbeddingBox>()
    private let lock = NSLock()

    var isAvailable: Bool { embedding != nil }

    enum CachePolicy { case store, noStore }

    private init() {
        cache.countLimit = 512
    }

    func vector(for text: String, policy: CachePolicy = .store) -> Entry? {
        guard let embedding else { return nil }
        let key = text as NSString

        if policy == .store, let cached = cache.object(forKey: key) {
            return cached.value
        }

        lock.lock()
        defer { lock.unlock() }

        if policy == .store, let cached = cache.object(forKey: key) {
            return cached.value
        }

        guard let v = embedding.vector(for: text) else { return nil }
        let floats = v.map(Float.init)
        let entry = Entry(vector: floats, norm: VectorMath.l2Norm(floats))

        if policy == .store {
            cache.setObject(EmbeddingBox(entry), forKey: key)
        }

        return entry
    }

    func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAllObjects()
    }
}

private final class EmbeddingBox: NSObject {
    let value: SentenceEmbeddingCache.Entry
    init(_ value: SentenceEmbeddingCache.Entry) { self.value = value }
}

enum VectorMath {
    static func l2Norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sqrt(sum)
    }

    static func cosine(_ a: [Float], _ b: [Float], normA: Float, normB: Float) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let denom = normA * normB
        guard denom > 0 else { return 0 }

        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot / denom
    }
}
```

This code is boring on purpose. Boring infrastructure code is good.

---

## Step 3: Building the search index

For small databases like mine (a few hundred items), you don't need complex infrastructure:

- Turn the query into an embedding once
- For each document:
  - Calculate how similar it is (cosine similarity)
  - Boost it if certain tags match
  - Boost it slightly if the category matches
- Sort by score, take the top results

Here's the code I use. You can drop this into any Swift project:

```swift
import Foundation

struct SemanticDoc<ID: Hashable> {
    let id: ID
    let text: String
    let tags: [String]
    let category: String?
}

final class SemanticIndex<ID: Hashable>: @unchecked Sendable {
    private let cache: SentenceEmbeddingCache
    private let docs: [SemanticDoc<ID>]

    private let docEmbeddings: [(v: [Float], n: Float)?]
    private let tagsLower: [[String]]

    init(docs: [SemanticDoc<ID>], cache: SentenceEmbeddingCache = .shared) {
        self.cache = cache
        self.docs = docs

        // Precompute embeddings once. Avoid `lazy` here if you'll call from tasks.
        if cache.isAvailable {
            var out: [(v: [Float], n: Float)?] = []
            out.reserveCapacity(docs.count)
            for d in docs {
                if let e = cache.vector(for: d.text, policy: .noStore) {
                    out.append((e.vector, e.norm))
                } else {
                    out.append(nil)
                }
            }
            self.docEmbeddings = out
        } else {
            self.docEmbeddings = Array(repeating: nil, count: docs.count)
        }

        self.tagsLower = docs.map { $0.tags.map { $0.lowercased() } }
    }

    func search(
        query: String,
        keywords: [String] = [],
        category: String? = nil,
        topK: Int = 3
    ) -> [SemanticDoc<ID>] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }

        let qLower = q.lowercased()
        let keywordSet = Set(keywords.map { $0.lowercased() })

        // Fallback when embedding is unavailable.
        guard let qEmbed = cache.vector(for: q) else {
            return docs
                .enumerated()
                .map { i, d in (d, tagScore(tagsLower[i], keywordSet, qLower, category, d.category)) }
                .sorted { $0.1 > $1.1 }
                .prefix(topK)
                .map { $0.0 }
        }

        var scored: [(SemanticDoc<ID>, Float)] = []
        scored.reserveCapacity(docs.count)

        for i in docs.indices {
            let boost = tagScore(tagsLower[i], keywordSet, qLower, category, docs[i].category)
            guard let dv = docEmbeddings[i] else {
                // Keep docs without embeddings in the candidate set via lexical score only.
                scored.append((docs[i], boost))
                continue
            }

            let sim = VectorMath.cosine(qEmbed.vector, dv.v, normA: qEmbed.norm, normB: dv.n)
            scored.append((docs[i], sim + boost))
        }

        return scored
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { $0.0 }
    }

    private func tagScore(
        _ tags: [String],
        _ keywordSet: Set<String>,
        _ queryLower: String,
        _ queryCategory: String?,
        _ docCategory: String?
    ) -> Float {
        var s: Float = 0

        let tokens = Set(
            queryLower.split { !$0.isLetter && !$0.isNumber }
                .map { String($0) }
        )

        // Tag boost: cheap lexical hints that help retrieval feel "snappier".
        for t in tags {
            if t.contains(" ") {
                if queryLower.contains(t) { s += 0.06 }
            } else if tokens.contains(t) {
                s += 0.06
            }
            if keywordSet.contains(t) { s += 0.05 }
        }

        // Category bias: small nudge, never a hard gate.
        if let qc = queryCategory?.lowercased(),
           let dc = docCategory?.lowercased(),
           qc == dc {
            s += 0.08
        }

        return s
    }
}
```

The things happening quietly here:

- **Precomputing** everything once at startup keeps the search fast
- **Hybrid scoring** (combining semantic similarity with keyword matching) makes it feel more accurate
- **Category boost** helps but doesn't force results into the wrong category

---

## Step 4: Extracting better keywords with NLTagger

Semantic search is useful, but people sometimes use very specific words you still want to catch.

I use `NLTagger` to pull out the important words (and their "root" forms):

```swift
import NaturalLanguage

func extractKeywords(_ text: String) -> [String] {
    let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma])
    tagger.string = text

    let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation, .joinNames]
    var counts: [String: Int] = [:]

    tagger.enumerateTags(in: text.startIndex..<text.endIndex,
                         unit: .word,
                         scheme: .lexicalClass,
                         options: options) { tag, range in
        guard let tag else { return true }
        guard tag == .noun || tag == .verb || tag == .adjective else { return true }

        let raw = String(text[range]).lowercased()
        guard raw.count >= 3 else { return true }

        let (lemma, _) = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lemma)
        let key = (lemma?.rawValue ?? raw).lowercased()

        counts[key, default: 0] += 1
        return true
    }

    return counts
        .sorted { $0.value > $1.value }
        .map { $0.key }
        .prefix(8)
        .map { $0 }
}
```

This extractor is an English-first heuristic. For Chinese text, you'll usually need extra tokenization or custom rules.

If you want to improve this, add a stopword list (for example, filtering out "the", "a", and "is"). The structure stays the same.

---

## Step 5: Putting it all together

Here's what it looks like in practice:

```swift
struct Analysis {
    let cleanedQuery: String
    let keywords: [String]
    let category: String
}

func analyzeQuestion(_ q: String) -> Analysis {
    let cleaned = q.trimmingCharacters(in: .whitespacesAndNewlines)
    let keywords = extractKeywords(cleaned)
    let category = "general"
    return Analysis(cleanedQuery: cleaned, keywords: keywords, category: category)
}

let frames = SemanticIndex(docs: [
    SemanticDoc(id: "reach-out-ex",
                text: "Should I text my ex?",
                tags: ["text", "ex", "message", "contact"],
                category: "love"),
    SemanticDoc(id: "accept-offer",
                text: "Should I accept this job offer?",
                tags: ["offer", "job", "accept", "career"],
                category: "career"),
])

let advice = SemanticIndex(docs: [
    SemanticDoc(id: "decision-deadline",
                text: "Set a day to decide. Indecision leaks time; choose with what you know, then adjust with what you learn.",
                tags: ["stuck", "delay", "decide"],
                category: "decision"),
])

let a = analyzeQuestion("I keep thinking about texting my ex. Should I do it?")
let bestFrame = frames.search(query: a.cleanedQuery, keywords: a.keywords, category: a.category, topK: 1).first
let bestAdvice = advice.search(query: a.cleanedQuery, keywords: a.keywords, category: a.category, topK: 2)
```

Notice what's NOT happening:

- No network request
- No giant AI model
- No black box magic

You can make something that feels smart with just a small database and clean code.

---

## Random things I learned

1. **Warm up the embeddings early**  
   The first time you use embeddings it's slow because of lazy initialization. In Wen Gua I start loading them during the card animation so the app feels smooth.

   ```swift
   Task.detached(priority: .utility) {
       _ = SentenceEmbeddingCache.shared.vector(for: "warm up")
   }
   ```

2. **Don't use `lazy var` if you're using it from background tasks**  
   I ran into weird crashes until I figured this out. Just compute everything in `init()`.

3. **Make your app work even when embeddings fail**  
   Sometimes they are not available. Your app should still do something useful.

4. **The quality of your search is limited by the quality of your writing**  
   Semantic search can't rescue weak snippets. I rewrote mine five times.

5. **Have a way to clear the cache under memory pressure**  
   Add `clearAll()` and call it from your app's memory warning path so cached vectors can be dropped quickly.

---

# Making it work for everyone (accessibility stuff)

I thought my app looked calm and minimal. Then I tested it with VoiceOver and realized it was basically unusable.

Turns out:
- "Calm" can still exclude people
- "Beautiful animations" can make some people nauseous
- "Minimal UI" can be meaningless to screen readers

Accessibility isn't something you add at the end. It's structure.

My approach in Wen Gua:

- One state machine (same logic, same steps)
- Two ways to show it:
  - Rich animations for people who want them
  - A VoiceOver/Reduce Motion friendly version that keeps the meaning without the motion

---

## Reading accessibility settings in SwiftUI

```swift
import SwiftUI

struct RitualView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled

    var isA11yMode: Bool { reduceMotion || voiceOverEnabled }

    var body: some View {
        Text(isA11yMode ? "Accessible path" : "Default path")
    }
}
```

The key idea: adapt the UI, don't fork it into two completely separate versions.

---

## A helper for animations that respect Reduce Motion

Reduce Motion doesn't mean "zero animation." It means "avoid unnecessary movement."

Gentle opacity fades are usually fine. Big spinning 3D rotations are not.

```swift
import SwiftUI

private struct AppAnimationModifier<Value: Equatable>: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let animation: Animation?
    let reducedAnimation: Animation?
    let value: Value

    func body(content: Content) -> some View {
        content.animation(reduceMotion ? reducedAnimation : animation, value: value)
    }
}

extension View {
    func appAnimation<V: Equatable>(
        _ animation: Animation?,
        reducedAnimation: Animation? = nil,
        value: V
    ) -> some View {
        modifier(AppAnimationModifier(animation: animation, reducedAnimation: reducedAnimation, value: value))
    }
}
```

Use it this way:

```swift
struct RevealExample: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealed = false

    var body: some View {
        Text("The lines are drawn")
            .opacity(revealed ? 1 : 0)
            .transition(reduceMotion ? .opacity : .scale.combined(with: .opacity))
            .appAnimation(.easeInOut(duration: 0.3),
                          reducedAnimation: .easeInOut(duration: 0.2),
                          value: revealed)
            .onAppear {
                withAnimation {
                    revealed = true
                }
            }
    }
}
```

---

## Hiding decorative stuff from VoiceOver

If something is purely visual—glows, particle effects, background canvases—hide it:

```swift
Canvas { context, size in
    // Decorative drawing
}
.accessibilityHidden(true)
```

Then make the semantic info clear with labels, hints, and logical focus order.

---

## Announcing key moments (VoiceOver is a conversation)

Animations show progress visually. VoiceOver needs an equivalent.

```swift
import UIKit

enum A11y {
    static func announce(_ message: String) {
        DispatchQueue.main.async {
            UIAccessibility.post(notification: .announcement, argument: message)
        }
    }
}
```

When the ritual reaches the reveal:

```swift
if voiceOverEnabled {
    A11y.announce("The lines are drawn")
}
```

Keep it short. Let it breathe.

---

## One ritual, two ways to show it

This is what I do in Wen Gua's card animation:

- If Motion is on: show the full card stack animation
- If Reduce Motion or VoiceOver is on:
  - Still do the same steps internally
  - Show the final card with a simple fade
  - Announce it's done

```swift
import SwiftUI

struct CardRevealView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled

    @State private var revealed = false
    @AccessibilityFocusState private var focusFinalCard: Bool

    var body: some View {
        ZStack {
            AnimatedStack()
                .accessibilityHidden(true)
                .opacity((reduceMotion || voiceOverEnabled) ? 0 : 1)

            FinalCard()
                .opacity((reduceMotion || voiceOverEnabled) ? (revealed ? 1 : 0) : 0)
                .accessibilityLabel("Hexagram card")
                .accessibilityFocused($focusFinalCard)
        }
        .onAppear { run() }
    }

    private func run() {
        if reduceMotion || voiceOverEnabled {
            withAnimation(.easeInOut(duration: 0.22)) {
                revealed = true
            }
            if voiceOverEnabled {
                focusFinalCard = true
                A11y.announce("The lines are drawn")
            }
        } else {
            // Run the rich animation path.
        }
    }
}

private struct AnimatedStack: View { var body: some View { Color.clear } }
private struct FinalCard: View { var body: some View { Color.clear } }
```

The important part isn't the visuals.  
It's telling **the same story in a way the person can actually receive.**

---

## Being honest about how it works

I added a "How it works" screen to Wen Gua that shows the whole pipeline: parsing, embedding, retrieval, formatting.

People deserve to know what happens to their words.

---

# License

- Code: MIT
- Writing: CC BY 4.0

---

## Why I care about this

"On-device intelligence" gets marketed as a feature.

I think of it more as basic decency.

The private thoughts people type into my app aren't training data.  
Their accessibility settings aren't inconveniences.  
Their attention isn't a resource to extract.

I've been coding since I was little, but music taught me something code tutorials didn't: the space between notes matters as much as the notes themselves.

Wen Gua is that space. A place to breathe before deciding. A ritual that respects the I Ching without gatekeeping it. A reminder that not everything needs to be in the cloud.

I spent months rebuilding this app to make it fully accessible and completely offline because I think your doubts deserve privacy and your decisions deserve dignity.

---

# 简体中文

## 为什么我不想把日记传到云端

排练前会发生一个很奇怪的事。我能在那坐二十分钟,脑子里一直转:

*如果我说节奏有问题,会不会让大家觉得我很烦?  
如果我不说,会不会更后悔?*

过度思考不是因为笨。是因为想法太多了,没地方放。

所以我做了 **Wen Gua(问卦)**。你输入困扰,抽个卦,然后看到一段文字——不是预测,是帮你把混乱理清楚。整个过程三分钟左右:够长到有用,够短到不会变成另一种逃避。

做这个的时候我发现两件事必须做到:

1. **你打的字要留在你手机里。**  
   想想看,如果一个帮你思考问题的app把你的困扰上传到服务器,这不是很奇怪吗?
2. **要对所有人友好。**  
   我第一版做完发现VoiceOver用户根本用不了,那时候才意识到这个问题。

这篇文章是我做这两个最难部分的笔记:

- 用 Apple 的 `NaturalLanguage` **让搜索离线工作**
- 让 UI **真的能用** (VoiceOver 和 Reduce Motion)

没有服务器,没有账号,没有追踪。就是你的手机在干活。

---

## "语义搜索"是什么

"语义搜索"听起来很高级,其实就是:

> 如果两句话意思差不多,app应该知道它们有关系——哪怕用的词完全不一样。

普通搜索找的是字。语义搜索找的是意思。

实现方法是用 **embedding**(嵌入向量)——把文字变成一串数字,如果两段文字意思接近,它们的数字就会靠得很近。

我知道这听起来很抽象。你可以想象成坐标,但不是地图上的坐标,是"意思的坐标"。

Wen Gua 只需要搜索我写的几百条短文本。所以不需要复杂的东西。就是:

- 数据库要小
- 文本要写得好
- 搜索代码要简单
- 全程离线

---

## NaturalLanguage 和 NLEmbedding 是干什么的

Apple 的 **NaturalLanguage** 框架给你一堆处理文本的工具:识别语言、分词、词性标注这些。

我需要的是 `NLEmbedding`,它把文字变成那些"数字坐标"。主要用这两个:

- `NLEmbedding.sentenceEmbedding(for:)` 处理整句话
- `NLEmbedding.wordEmbedding(for:)` 处理单个词

有个坑我踩了很久: 这些 embedding 不是一直能用的。有些语言或设备上就是没有。所以代码要能处理这种情况。

---

## 我怎么做的

有人把这种叫 "RAG"(检索增强生成)。但 Wen Gua 里没有大模型从零写文字。所有内容都是我自己写的,app 只是找出最合适的。

我其实更喜欢这样。感觉更诚实。

流程是这样:

1. **Parse** 清理用户输入,让它更好搜
2. **Sense** 识别语气(这样回复能更贴切)
3. **Embed** 把问题变成数字
4. **Retrieve** 找最匹配的内容
5. **Compose** 组成完整的回答
6. **Format** 排版,确保 VoiceOver 能读

---

## 第一步:获取 embedding

```swift
import NaturalLanguage

func sentenceVector(for text: String) -> [Double]? {
    guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
        return nil
    }
    return embedding.vector(for: text)
}
```

两个我花了很久才搞懂的事:

- `NLEmbedding.sentenceEmbedding(for:)` 可能就是不行。返回 `nil` 是正常的。
- 返回值是 `[Double]?` 但我发现转成 `Float` 能省内存,算得也快。

---

## 第二步:缓存 embedding(这让速度快了很多)

我一开始一直在重复计算同样的 embedding。超级慢。

后来我学会了缓存:

- 把 query embedding 放在 `NSCache` 里
- 同时存数字和"长度"(L2 范数)
- 确保线程安全,因为 Swift 对这个很严格

```swift
import Foundation
import NaturalLanguage

/// Thread-safe, shared embedding cache.
final class SentenceEmbeddingCache: @unchecked Sendable {
    static let shared = SentenceEmbeddingCache()

    struct Entry {
        let vector: [Float]
        let norm: Float
    }

    private let embedding = NLEmbedding.sentenceEmbedding(for: .english)
    private let cache = NSCache<NSString, EmbeddingBox>()
    private let lock = NSLock()

    var isAvailable: Bool { embedding != nil }

    enum CachePolicy { case store, noStore }

    private init() {
        cache.countLimit = 512
    }

    func vector(for text: String, policy: CachePolicy = .store) -> Entry? {
        guard let embedding else { return nil }
        let key = text as NSString

        if policy == .store, let cached = cache.object(forKey: key) {
            return cached.value
        }

        lock.lock()
        defer { lock.unlock() }

        if policy == .store, let cached = cache.object(forKey: key) {
            return cached.value
        }

        guard let v = embedding.vector(for: text) else { return nil }
        let floats = v.map(Float.init)
        let entry = Entry(vector: floats, norm: VectorMath.l2Norm(floats))

        if policy == .store {
            cache.setObject(EmbeddingBox(entry), forKey: key)
        }

        return entry
    }

    func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAllObjects()
    }
}

private final class EmbeddingBox: NSObject {
    let value: SentenceEmbeddingCache.Entry
    init(_ value: SentenceEmbeddingCache.Entry) { self.value = value }
}

enum VectorMath {
    static func l2Norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sqrt(sum)
    }

    static func cosine(_ a: [Float], _ b: [Float], normA: Float, normB: Float) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let denom = normA * normB
        guard denom > 0 else { return 0 }

        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot / denom
    }
}
```

这段代码故意写得很无聊。基础设施代码无聊是好事。

---

## 第三步:搭建搜索索引

对于像我这样的小数据库(几百条),不需要复杂的东西。就是:

- query embedding 只算一次
- 对每个文档:
  - 算相似度(余弦相似度)
  - 如果 tag 匹配就加分
  - 如果类别匹配稍微加点分
- 排序,取前几个

这是我用的代码。你可以直接放到任何 Swift 项目里:

```swift
import Foundation

struct SemanticDoc<ID: Hashable> {
    let id: ID
    let text: String
    let tags: [String]
    let category: String?
}

final class SemanticIndex<ID: Hashable>: @unchecked Sendable {
    private let cache: SentenceEmbeddingCache
    private let docs: [SemanticDoc<ID>]

    private let docEmbeddings: [(v: [Float], n: Float)?]
    private let tagsLower: [[String]]

    init(docs: [SemanticDoc<ID>], cache: SentenceEmbeddingCache = .shared) {
        self.cache = cache
        self.docs = docs

        // Precompute embeddings once. Avoid `lazy` here if you'll call from tasks.
        if cache.isAvailable {
            var out: [(v: [Float], n: Float)?] = []
            out.reserveCapacity(docs.count)
            for d in docs {
                if let e = cache.vector(for: d.text, policy: .noStore) {
                    out.append((e.vector, e.norm))
                } else {
                    out.append(nil)
                }
            }
            self.docEmbeddings = out
        } else {
            self.docEmbeddings = Array(repeating: nil, count: docs.count)
        }

        self.tagsLower = docs.map { $0.tags.map { $0.lowercased() } }
    }

    func search(
        query: String,
        keywords: [String] = [],
        category: String? = nil,
        topK: Int = 3
    ) -> [SemanticDoc<ID>] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }

        let qLower = q.lowercased()
        let keywordSet = Set(keywords.map { $0.lowercased() })

        // Fallback when embedding is unavailable.
        guard let qEmbed = cache.vector(for: q) else {
            return docs
                .enumerated()
                .map { i, d in (d, tagScore(tagsLower[i], keywordSet, qLower, category, d.category)) }
                .sorted { $0.1 > $1.1 }
                .prefix(topK)
                .map { $0.0 }
        }

        var scored: [(SemanticDoc<ID>, Float)] = []
        scored.reserveCapacity(docs.count)

        for i in docs.indices {
            let boost = tagScore(tagsLower[i], keywordSet, qLower, category, docs[i].category)
            guard let dv = docEmbeddings[i] else {
                // Keep docs without embeddings in the candidate set via lexical score only.
                scored.append((docs[i], boost))
                continue
            }

            let sim = VectorMath.cosine(qEmbed.vector, dv.v, normA: qEmbed.norm, normB: dv.n)
            scored.append((docs[i], sim + boost))
        }

        return scored
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { $0.0 }
    }

    private func tagScore(
        _ tags: [String],
        _ keywordSet: Set<String>,
        _ queryLower: String,
        _ queryCategory: String?,
        _ docCategory: String?
    ) -> Float {
        var s: Float = 0

        let tokens = Set(
            queryLower.split { !$0.isLetter && !$0.isNumber }
                .map { String($0) }
        )

        // Tag boost: cheap lexical hints that help retrieval feel "snappier".
        for t in tags {
            if t.contains(" ") {
                if queryLower.contains(t) { s += 0.06 }
            } else if tokens.contains(t) {
                s += 0.06
            }
            if keywordSet.contains(t) { s += 0.05 }
        }

        // Category bias: small nudge, never a hard gate.
        if let qc = queryCategory?.lowercased(),
           let dc = docCategory?.lowercased(),
           qc == dc {
            s += 0.08
        }

        return s
    }
}
```

几个在默默工作的部分:

- **预计算** 启动时算一次,保持搜索快速
- **混合打分** (语义相似度 + 关键词匹配) 让结果更准
- **类别加分** 有帮助但不会强行塞结果

---

## 第四步:用 NLTagger 提取更好的关键词

语义搜索很厉害,但有时候人们用很具体的词,你也想抓住那些。

我用 `NLTagger` 提取重要的词(还有它们的"词根"):

```swift
import NaturalLanguage

func extractKeywords(_ text: String) -> [String] {
    let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma])
    tagger.string = text

    let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation, .joinNames]
    var counts: [String: Int] = [:]

    tagger.enumerateTags(in: text.startIndex..<text.endIndex,
                         unit: .word,
                         scheme: .lexicalClass,
                         options: options) { tag, range in
        guard let tag else { return true }
        guard tag == .noun || tag == .verb || tag == .adjective else { return true }

        let raw = String(text[range]).lowercased()
        guard raw.count >= 3 else { return true }

        let (lemma, _) = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lemma)
        let key = (lemma?.rawValue ?? raw).lowercased()

        counts[key, default: 0] += 1
        return true
    }

    return counts
        .sorted { $0.value > $1.value }
        .map { $0.key }
        .prefix(8)
        .map { $0 }
}
```

这个 `extractKeywords` 本质是英文优先的启发式,对中文提词通常会比较弱,需要额外分词或规则。

要做得更好的话,加个停用词表(过滤掉"的"、"是"这些)。但整体结构不会变。

---

## 第五步:把它拼起来

实际用起来长这样:

```swift
struct Analysis {
    let cleanedQuery: String
    let keywords: [String]
    let category: String
}

func analyzeQuestion(_ q: String) -> Analysis {
    let cleaned = q.trimmingCharacters(in: .whitespacesAndNewlines)
    let keywords = extractKeywords(cleaned)
    let category = "general"
    return Analysis(cleanedQuery: cleaned, keywords: keywords, category: category)
}

let frames = SemanticIndex(docs: [
    SemanticDoc(id: "reach-out-ex",
                text: "Should I text my ex?",
                tags: ["text", "ex", "message", "contact"],
                category: "love"),
    SemanticDoc(id: "accept-offer",
                text: "Should I accept this job offer?",
                tags: ["offer", "job", "accept", "career"],
                category: "career"),
])

let advice = SemanticIndex(docs: [
    SemanticDoc(id: "decision-deadline",
                text: "Set a day to decide. Indecision leaks time; choose with what you know, then adjust with what you learn.",
                tags: ["stuck", "delay", "decide"],
                category: "decision"),
])

let a = analyzeQuestion("I keep thinking about texting my ex. Should I do it?")
let bestFrame = frames.search(query: a.cleanedQuery, keywords: a.keywords, category: a.category, topK: 1).first
let bestAdvice = advice.search(query: a.cleanedQuery, keywords: a.keywords, category: a.category, topK: 2)
```

注意这里:

- 没有网络请求
- 没有巨大模型
- 没有任何无法解释的黑箱

你仍然可以做出"看起来很聪明"的体验——只要语料精炼,打分朴素,结构认真。

---

## 我踩的坑

1. **提前预热 embedding**  
   第一次用 embedding 会很慢因为懒加载。我在卡片动画时就开始加载,这样 app 用起来很流畅。

   ```swift
   Task.detached(priority: .utility) {
       _ = SentenceEmbeddingCache.shared.vector(for: "warm up")
   }
   ```

2. **别用 `lazy var` 如果你要在后台任务里用**  
   我遇到了奇怪的崩溃直到搞懂这个。就在 `init()` 里算好所有东西。

3. **embedding 失败时 app 也要能用**  
   有时候它们就是不可用。你的 app 还是要做点有用的事。

4. **搜索质量取决于你写的内容质量**  
   语义搜索救不了烂文本。我的内容改了五遍。

5. **内存紧张时要有清缓存出口**  
   给缓存加 `clearAll()`，并在 app 的 memory warning 路径里调用，能快速释放缓存向量。

---

# 让所有人都能用(无障碍部分)

我以为我的 app 看起来很安静很简洁。然后我用 VoiceOver 测了一下,发现基本用不了。

结果发现:
- "安静"也可能排除别人
- "漂亮的动画"可能让有些人恶心
- "极简 UI"对屏幕阅读器可能毫无意义

无障碍不是最后加上去的东西。它是结构。

我在 Wen Gua 里的做法:

- 一个状态机(同一套逻辑,同一组步骤)
- 两种展示方式:
  - 想要的人有丰富动画
  - VoiceOver/Reduce Motion 友好的版本,保留意义但减少运动

---

## 在 SwiftUI 里读取无障碍设置

```swift
import SwiftUI

struct RitualView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled

    var isA11yMode: Bool { reduceMotion || voiceOverEnabled }

    var body: some View {
        Text(isA11yMode ? "Accessible path" : "Default path")
    }
}
```

核心思路:让 UI 自适应,别做两套完全不同的界面。

---

## 一个尊重 Reduce Motion 的动画辅助工具

Reduce Motion 不是说"零动画"。是说"避免不必要的运动"。

轻微的透明度过渡通常没问题。大幅旋转的 3D 效果就不行。

```swift
import SwiftUI

private struct AppAnimationModifier<Value: Equatable>: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let animation: Animation?
    let reducedAnimation: Animation?
    let value: Value

    func body(content: Content) -> some View {
        content.animation(reduceMotion ? reducedAnimation : animation, value: value)
    }
}

extension View {
    func appAnimation<V: Equatable>(
        _ animation: Animation?,
        reducedAnimation: Animation? = nil,
        value: V
    ) -> some View {
        modifier(AppAnimationModifier(animation: animation, reducedAnimation: reducedAnimation, value: value))
    }
}
```

这样用:

```swift
struct RevealExample: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealed = false

    var body: some View {
        Text("The lines are drawn")
            .opacity(revealed ? 1 : 0)
            .transition(reduceMotion ? .opacity : .scale.combined(with: .opacity))
            .appAnimation(.easeInOut(duration: 0.3),
                          reducedAnimation: .easeInOut(duration: 0.2),
                          value: revealed)
            .onAppear {
                withAnimation {
                    revealed = true
                }
            }
    }
}
```

---

## 把纯装饰层从 VoiceOver 世界里"隐形"

光晕、粒子、Canvas 背景、3D 效果——如果它们不承载信息,就应该隐藏:

```swift
Canvas { context, size in
    // 纯装饰绘制
}
.accessibilityHidden(true)
```

然后把真正的语义信息用 `accessibilityLabel` / `Hint` / `combine` 的方式交代清楚。

---

## 关键时刻要"说出来"

动画会用视觉表达进度。VoiceOver 需要等价的信号。

```swift
import UIKit

enum A11y {
    static func announce(_ message: String) {
        DispatchQueue.main.async {
            UIAccessibility.post(notification: .announcement, argument: message)
        }
    }
}
```

在仪式完成的瞬间:

```swift
if voiceOverEnabled {
    A11y.announce("The lines are drawn")
}
```

句子要短,给听觉留空间。

---

## 一个可复用模式:一个仪式,两种揭示方式

这就是 Wen Gua 在卡牌动画里做的事:

- Motion 开启时:完整的卡堆动画
- Reduce Motion 或 VoiceOver 开启时:
  - 仍然推进同样的步骤
  - 用 crossfade 显示最终结果
  - 发布语音提示

```swift
import SwiftUI

struct CardRevealView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled

    @State private var revealed = false
    @AccessibilityFocusState private var focusFinalCard: Bool

    var body: some View {
        ZStack {
            AnimatedStack()
                .accessibilityHidden(true)
                .opacity((reduceMotion || voiceOverEnabled) ? 0 : 1)

            FinalCard()
                .opacity((reduceMotion || voiceOverEnabled) ? (revealed ? 1 : 0) : 0)
                .accessibilityLabel("Hexagram card")
                .accessibilityFocused($focusFinalCard)
        }
        .onAppear { run() }
    }

    private func run() {
        if reduceMotion || voiceOverEnabled {
            withAnimation(.easeInOut(duration: 0.22)) {
                revealed = true
            }
            if voiceOverEnabled {
                focusFinalCard = true
                A11y.announce("The lines are drawn")
            }
        } else {
            // 默认动画路径…
        }
    }
}

private struct AnimatedStack: View { var body: some View { Color.clear } }
private struct FinalCard: View { var body: some View { Color.clear } }
```

最重要的不是视觉,而是完整性:  
**同一个故事,用用户能够接收的方式讲出来。**

---

## 诚实地展示它怎么工作的

我给 Wen Gua 加了个"工作原理"页面,展示整个流程:解析、嵌入、检索、格式化。

让用户知道"他们的文字发生了什么",是一种尊重。

---

# 许可协议

- 代码:MIT
- 文档:CC BY 4.0

---

## 我为什么在意这个

"端侧智能"有时被当作卖点。

我更愿意把它当作...做人该做的?

人们在我 app 里打的私人想法不是训练数据。  
他们的无障碍设置不是麻烦。  
他们的注意力不是矿产。

我从小就在写代码,但音乐教会了我一些代码教程没教的东西:音符之间的停顿和音符本身一样重要。

Wen Gua 就是那个停顿。决定前呼吸的地方。一个尊重易经但不搞门槛的仪式。一个提醒——不是所有东西都要在云端。

我花了好几个月重写这个 app 让它完全无障碍、完全离线,因为我觉得你的疑虑值得隐私,你的决定值得尊严。
