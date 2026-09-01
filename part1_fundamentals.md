# Learn to Code from Scratch — Part 1: Fundamentals

> [!NOTE]
> This guide uses TypeScript. Everything here applies to JavaScript too — TypeScript just adds type safety on top.

---

## 1. What Is Programming?

Programming = giving a computer **step-by-step instructions**. That's it.

Think of it like a recipe:
- "Take 2 eggs" → get data
- "Beat them" → process data  
- "Pour into pan" → output result

A computer is **extremely literal**. It does exactly what you say — nothing more, nothing less.

---

## 2. Variables — Storing Information

A variable is a **labelled box** that holds a value.

```typescript
// Declaring variables
let age = 25              // can change later
const name = "Prajwal"    // cannot change (constant)
var old = "don't use this" // old way, avoid it

// Changing a variable
age = 26        // ✅ works — let allows reassignment
name = "John"   // ❌ ERROR — const cannot be reassigned
```

### When to use `let` vs `const`?
- **`const`** by default (90% of the time). If the value won't change, lock it down.
- **`let`** only when the value WILL change (counters, accumulators, flags).

---

## 3. Data Types — What Can Go Inside the Box?

```typescript
// Primitive types (simple, single values)
const age: number = 25                  // numbers (integer or decimal)
const name: string = "Prajwal"          // text
const isActive: boolean = true          // true or false
const nothing: null = null              // intentionally empty
const notSet: undefined = undefined     // not yet assigned

// Complex types
const scores: number[] = [85, 92, 78]           // array (list of items)
const person: { name: string; age: number } = {  // object (group of properties)
  name: "Prajwal",
  age: 25
}
```

### The TypeScript Difference

In plain JavaScript, you can put anything anywhere:
```javascript
let x = 5       // number
x = "hello"     // now it's a string — JS doesn't care
```

TypeScript **catches mistakes early**:
```typescript
let x: number = 5
x = "hello"     // ❌ ERROR: Type 'string' is not assignable to type 'number'
```

This saves you from bugs that would only appear when users run your code.

---

## 4. Operators — Doing Things With Data

```typescript
// Arithmetic
5 + 3    // 8        (addition)
10 - 4   // 6        (subtraction)
3 * 7    // 21       (multiplication)
15 / 4   // 3.75     (division)
15 % 4   // 3        (remainder/modulo — very useful!)

// Comparison (returns true or false)
5 === 5    // true   (strict equal — ALWAYS use this)
5 !== 3    // true   (not equal)
5 > 3      // true   (greater than)
5 < 3      // false  (less than)
5 >= 5     // true   (greater or equal)

// ⚠️ NEVER use == (loose equal). It does weird conversions.
"5" == 5   // true  — BAD, misleading
"5" === 5  // false — GOOD, honest

// Logical (combine conditions)
true && true    // true   (AND — both must be true)
true || false   // true   (OR — at least one must be true)
!true           // false  (NOT — flips it)
```

---

## 5. Conditionals — Making Decisions

This is where **logic** starts. Your code makes choices.

```typescript
const temperature = 35

// Simple if-else
if (temperature > 30) {
  console.log("It's hot!")
} else if (temperature > 20) {
  console.log("It's warm")
} else {
  console.log("It's cold")
}
// Output: "It's hot!"
```

### How to Think About Conditionals

Ask yourself: **"What are the possible scenarios?"**

```
Scenario 1: temp > 30  → hot
Scenario 2: temp > 20  → warm
Scenario 3: everything else → cold
```

Order matters! Conditions are checked **top to bottom**. First match wins.

### Real Example — Heatmap Score (from your project)

```typescript
const heatmapScore = 75

let color: string
if (heatmapScore > 70) {
  color = "red"       // high risk
} else if (heatmapScore >= 40) {
  color = "amber"     // medium risk
} else {
  color = "green"     // low risk
}
// color = "red"
```

### Ternary — One-Line If/Else

```typescript
// Long way
let label: string
if (age >= 18) {
  label = "adult"
} else {
  label = "minor"
}

// Short way (ternary)
const label = age >= 18 ? "adult" : "minor"
```

---

## 6. Loops — Repeating Actions

### For Loop — When You Know How Many Times

```typescript
// Print 1 to 5
for (let i = 1; i <= 5; i++) {
  console.log(i)
}
// 1, 2, 3, 4, 5

// Breaking it down:
// let i = 1      → start at 1
// i <= 5         → keep going while i is ≤ 5
// i++            → add 1 to i after each round (i++ is shorthand for i = i + 1)
```

### For...of — Loop Through a List

```typescript
const fruits = ["apple", "banana", "cherry"]

for (const fruit of fruits) {
  console.log(fruit)
}
// "apple", "banana", "cherry"
```

### While Loop — When You Don't Know How Many Times

```typescript
let count = 0
while (count < 3) {
  console.log(`Count is ${count}`)
  count++
}
// "Count is 0", "Count is 1", "Count is 2"
```

### When to Use Which?

| Loop | Use When |
|---|---|
| `for` | You know the exact count |
| `for...of` | You're iterating over a list |
| `while` | You loop until a condition changes |
| `.forEach()` | You want a functional style (covered later) |

---

## 7. Functions — Reusable Blocks of Logic

A function is a **named recipe** you can call whenever you need it.

```typescript
// Defining a function
function greet(name: string): string {
  return `Hello, ${name}!`
}

// Calling it
const message = greet("Prajwal")  // "Hello, Prajwal!"
const message2 = greet("Sajad")   // "Hello, Sajad!"
```

### Anatomy of a Function

```typescript
function calculateTax(income: number, rate: number): number {
//       ^ name        ^ parameters (inputs)         ^ return type
  const tax = income * rate
  return tax    // ← what comes back out
}

const myTax = calculateTax(50000, 0.3)  // 15000
```

### Arrow Functions (Modern Style)

```typescript
// Regular function
function add(a: number, b: number): number {
  return a + b
}

// Arrow function (same thing, shorter)
const add = (a: number, b: number): number => {
  return a + b
}

// Ultra-short (single expression — implicit return)
const add = (a: number, b: number): number => a + b
```

### Functions That Don't Return Anything

```typescript
function logMessage(msg: string): void {
  console.log(msg)
  // no return — void means "returns nothing"
}
```

---

## 8. Arrays — Working With Lists of Data

```typescript
const scores = [85, 92, 78, 96, 88]

// Access by index (0-based!)
scores[0]    // 85 (first item)
scores[4]    // 88 (fifth item)
scores.length // 5

// Add and remove
scores.push(100)      // add to end → [85, 92, 78, 96, 88, 100]
scores.pop()           // remove from end → [85, 92, 78, 96, 88]
```

### The Big Three: map, filter, reduce

These are the most important array methods. Master them.

```typescript
const numbers = [1, 2, 3, 4, 5]

// MAP — transform every item (returns new array, same length)
const doubled = numbers.map(n => n * 2)
// [2, 4, 6, 8, 10]

// FILTER — keep items that match a condition (returns new array, shorter or equal)
const evens = numbers.filter(n => n % 2 === 0)
// [2, 4]

// REDUCE — combine all items into one value
const sum = numbers.reduce((total, n) => total + n, 0)
// 15 (0 + 1 + 2 + 3 + 4 + 5)
```

### Real Example — Calculating Weighted Exposure

```typescript
interface Client {
  name: string
  commission: number
  heatmapScore: number
}

const clients: Client[] = [
  { name: "Client A", commission: 5000, heatmapScore: 85 },
  { name: "Client B", commission: 3000, heatmapScore: 55 },
  { name: "Client C", commission: 8000, heatmapScore: 20 },
]

// Step 1: Calculate weight based on heatmap score
function getWeight(score: number): number {
  if (score > 70) return 1.0
  if (score >= 40) return 0.5
  return 0.2
}

// Step 2: Calculate total weighted exposure
const totalExposure = clients.reduce((sum, client) => {
  const weight = getWeight(client.heatmapScore)
  return sum + (client.commission * weight)
}, 0)
// 5000*1.0 + 3000*0.5 + 8000*0.2 = 5000 + 1500 + 1600 = 8100
```

---

## 9. Objects — Grouping Related Data

```typescript
// Simple object
const broker = {
  name: "Prajwal",
  email: "praj@example.com",
  clientCount: 47,
  isActive: true,
}

// Access properties
broker.name          // "Prajwal"
broker["email"]      // "praj@example.com" (bracket notation)

// Destructuring — unpack properties into variables
const { name, clientCount } = broker
console.log(name)         // "Prajwal"
console.log(clientCount)  // 47
```

### Interfaces — Defining the Shape of Objects

```typescript
// An interface is a blueprint
interface BrokerProfile {
  id: string
  name: string
  email: string
  isActive: boolean
  clientCount: number
  plan?: string  // ? means optional — can be undefined
}

// Now TypeScript enforces the shape
const broker: BrokerProfile = {
  id: "abc-123",
  name: "Prajwal",
  email: "praj@example.com",
  isActive: true,
  clientCount: 47,
  // plan is optional, so we can skip it
}

// If you miss a required field:
const bad: BrokerProfile = {
  id: "abc",
  name: "Test",
  // ❌ ERROR: Property 'email' is missing
}
```

---

## 10. Logic Building — How to Think Like a Programmer

> [!IMPORTANT]
> This is the most important section. Syntax you can look up. **Thinking** is the real skill.

### The 4-Step Process

**Step 1: Understand the problem in plain English**
> "I need to find clients whose clawback window expires in the next 30 days AND have an open alert."

**Step 2: Break it into smaller steps**
> 1. Get all clients
> 2. Filter to those with ≤30 days remaining
> 3. Of those, filter to ones with open alerts
> 4. Sort by urgency

**Step 3: Write pseudocode (fake code in English)**
```
get all clients for this broker
for each client:
  if daysRemaining <= 30:
    check if client has open T1 or T2 alert
    if yes: add to urgent list
sort urgent list by daysRemaining ascending
return urgent list
```

**Step 4: Translate to real code**
```typescript
function getUrgentClients(clients: Client[], alerts: Alert[]): Client[] {
  return clients
    .filter(c => c.daysRemaining <= 30)
    .filter(c => alerts.some(a => 
      a.clientId === c.id && 
      (a.tier === 'T1' || a.tier === 'T2') && 
      a.status === 'open'
    ))
    .sort((a, b) => a.daysRemaining - b.daysRemaining)
}
```

### Common Logic Patterns

#### Pattern 1: Accumulator (building up a result)
```typescript
// Count how many alerts were dismissed
let dismissedCount = 0
for (const alert of alerts) {
  if (alert.outcome === 'false_positive') {
    dismissedCount++
  }
}
// Or with filter:
const dismissedCount = alerts.filter(a => a.outcome === 'false_positive').length
```

#### Pattern 2: Lookup / Search
```typescript
// Find a specific client
const targetClient = clients.find(c => c.id === "abc-123")
// returns the client object, or undefined if not found

// Check if something exists
const hasHighRisk = clients.some(c => c.heatmapScore > 70)
// returns true/false
```

#### Pattern 3: Transform Data
```typescript
// Turn a list of clients into a summary
const summary = clients.map(c => ({
  label: `Client #${c.id}`,
  risk: c.heatmapScore > 70 ? "HIGH" : "LOW",
  commission: `$${c.commission.toLocaleString()}`
}))
```

#### Pattern 4: Group By
```typescript
// Group clients by month
const byMonth: Record<string, Client[]> = {}

for (const client of clients) {
  const month = client.expiryDate.slice(0, 7) // "2026-05"
  if (!byMonth[month]) {
    byMonth[month] = []  // create empty array if first time
  }
  byMonth[month].push(client)
}
// { "2026-05": [client1, client2], "2026-06": [client3] }
```

#### Pattern 5: Early Return (guard clauses)
```typescript
// ❌ Deeply nested — hard to read
function processAlert(alert: Alert) {
  if (alert) {
    if (alert.status === 'open') {
      if (alert.confidence > 0.65) {
        // actually do the work
      }
    }
  }
}

// ✅ Guard clauses — flat and clear
function processAlert(alert: Alert) {
  if (!alert) return
  if (alert.status !== 'open') return
  if (alert.confidence <= 0.65) return

  // actually do the work — only valid cases reach here
}
```

---

## Exercises — Try These Yourself

### Exercise 1: Basic
Write a function that takes a number and returns "Fizz" if divisible by 3, "Buzz" if divisible by 5, "FizzBuzz" if both, or the number as a string.

### Exercise 2: Arrays
Given `[10, 25, 30, 45, 50, 65]`, write code to:
- Filter numbers greater than 30
- Double each remaining number
- Sum the result

### Exercise 3: Logic Building
Write a function `classifyRisk(score: number, daysLeft: number): string` that returns:
- "CRITICAL" if score > 70 AND daysLeft < 30
- "HIGH" if score > 70 OR daysLeft < 30
- "MEDIUM" if score > 40
- "LOW" otherwise

---

> [!TIP]
> **Next part covers:** Async/await, Promises, error handling, TypeScript generics, and how APIs work. Ask me for Part 2 when you're ready.
