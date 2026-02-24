# DocCompliance Pro - Unified Application Wireframe

## 🏗️ Overall Application Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                 DocCompliance Pro - Enterprise                  │
├─────────────┬───────────────────────────────────────────────────┤
│             │              Top Navigation & Search              │
│             ├───────────────────────────────────────────────────┤
│   Sidebar   │                                                   │
│             │                                                   │
│             │              Main Content Area                    │
│             │                                                   │
│             │                                                   │
│             │                                                   │
└─────────────┴───────────────────────────────────────────────────┘
```

---

## 📱 Main Navigation (Simplified ProjectFlow Style)

```
┌─────────────────┐
│ DocCompliance   │ ← Logo & Brand
│      Pro        │
├─────────────────┤
│ 🏠 Welcome      │ ← Main user experience
│ 📁 Projects     │ ← Compliance projects (primary workflow)
│ 📄 Documents    │ ← Document library & upload
│ ✅ Tasks        │ ← My tasks & assignments
│ 📊 Reports      │ ← Compliance reports & analytics
│ ❓ Help         │ ← Getting started & support
├─────────────────┤
│   ⚙️ Settings   │ ← ADVANCED FEATURES GROUPED HERE
│   └─ 🛡️ Compliance Hub    │ ← Full compliance tracking
│   └─ 🤖 AI & Automation   │ ← Agent management
│   └─ 📋 Policies & Templates │ ← Template management
│   └─ 🔧 System Admin      │ ← User management & logs
│   └─ 📈 Analytics         │ ← Advanced reporting
├─────────────────┤
│   Workspaces    │
│ 🎯 SOC 2        │ ← Quick project access
│ 🛡️ GDPR         │
│ 📊 ISO 27001    │
│ 🏥 HIPAA        │
└─────────────────┘
```

---

## 🔝 Top Navigation Bar

```
┌─────────────────────────────────────────────────────────────────┐
│ [🔍 Search projects, documents, tasks...] [📊 Status] [📋 Logs] [👤] [🔔] │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏠 Welcome Page (ProjectFlow Style with Compliance Context)

```
┌─────────────────────────────────────────────────────────────────┐
│ Good morning, John! 👋                                          │
│ Let's keep your compliance projects on track today              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────────┐  ┌─────────────────────┐               │
│ │   Start New         │  │   View All          │               │
│ │   Compliance        │  │   Projects          │               │
│ │   Project           │  │                     │               │
│ │   [+] Create...     │  │   [📁] Browse...    │               │
│ └─────────────────────┘  └─────────────────────┘               │
│                                                                 │
│ Quick Actions                                                   │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 📤 Upload   │ │ ✅ Quick    │ │ 📊 View     │               │
│ │ Documents   │ │ Task Entry  │ │ Dashboard   │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ ┌─────────────────────┐  ┌─────────────────────┐               │
│ │ 🕐 My Tasks (3)     │  │ 🎯 Active Projects  │               │
│ │ ┌─────────────────┐ │  │ ┌─────────────────┐ │               │
│ │ │ 🔴 Review GDPR  │ │  │ │ 🟢 SOC 2 (75%)  │ │               │
│ │ │    policy docs  │ │  │ │ 🟡 ISO (60%)    │ │               │
│ │ │    Due: Today   │ │  │ │ 🔴 HIPAA (30%)  │ │               │
│ │ │ ✅ Upload audit │ │  │ └─────────────────┘ │               │
│ │ │    files        │ │  └─────────────────────┘               │
│ │ │ 📋 Complete HR  │ │                                        │
│ │ │    checklist    │ │                                        │
│ │ └─────────────────┘ │                                        │
│ └─────────────────────┘                                        │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Compliance Health Overview          🛡️ Overall: 78% 🟡    │ │
│ │ 🟢 GDPR: 95%   🟢 SOX: 88%   🟡 ISO: 72%   🔴 HIPAA: 45% │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Need Help Getting Started?                                      │
│ [📖 Quick Start] [📹 Video Tour] [💬 Contact Support]          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Projects Page (Main Workflow Hub)

```
┌─────────────────────────────────────────────────────────────────┐
│ Compliance Projects                                             │
│ Manage your compliance initiatives and track progress           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────────┐  ┌─────────────────────┐               │
│ │   Create New        │  │   Project           │               │
│ │   Project           │  │   Templates         │               │
│ │   [+] Start new...  │  │   [📋] Browse...    │               │
│ └─────────────────────┘  └─────────────────────┘               │
│                                                                 │
│ Active Projects                    [Filter] [Sort] [View: List] │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Project Name    │Status │Progress│Compliance│Due Date │Actions│ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ SOC 2 Readiness │🟢 Active│  75%  │   92%   │Dec 30  │[📄][📊]│ │
│ │ GDPR Q1 Review  │🔄 Review│  90%  │   95%   │Dec 20  │[📄][📊]│ │
│ │ ISO 27001       │🟡 Risk  │  25%  │   60%   │Jan 15  │[📄][📊]│ │
│ │ HIPAA Audit     │🔴 Behind│  30%  │   45%   │Feb 28  │[📄][📊]│ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Quick Stats                                                     │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│ │Active   │ │Documents│ │AI Agents│ │Overall  │               │
│ │Projects │ │Processed│ │Working  │ │Health   │               │
│ │   4     │ │  247    │ │   3     │ │  78%    │               │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🚨 Attention Required                                       │ │
│ │ • HIPAA Audit - 30% complete, due in 8 weeks               │ │
│ │ • 3 documents need review in SOC 2 project                 │ │
│ │ • AI Agent found 2 compliance gaps in ISO project         │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 Documents Page (Enhanced with AI Features)

```
┌─────────────────────────────────────────────────────────────────┐
│ Document Management                                             │
│ Upload, manage, and let AI analyze your compliance documents    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 📤 Upload   │ │ 📁 Document │ │ 🤖 AI       │               │
│ │ Documents   │ │ Library     │ │ Processing  │               │
│ │ Smart upload│ │ Browse all  │ │ Queue: 3    │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 📤 Quick Upload - AI-Powered Document Processing           │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │          Drag & Drop Files Here                         │ │ │
│ │ │         or click to browse                              │ │ │
│ │ │  🤖 AI will automatically categorize and analyze       │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ │ Project: [Auto-assign ▼] Type: [Auto-detect ▼]             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Recent Documents          [🔍 Search] [Filter] [Sort: Date ↓]   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Name           │Type│Size │Status  │Compliance│Project │AI  │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Privacy 2024   │PDF │2.4MB│✅ Ready│   95%   │GDPR   │🤖✅│ │
│ │ Terms Service  │DOC │856KB│🔄 Review│   88%   │Legal  │🤖⚠️│ │
│ │ Data Agreement │PDF │1.2MB│⏳ Queue │   --    │GDPR   │🤖⏳│ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Statistics                                                      │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│ │Total    │ │AI       │ │Compliance│ │Storage  │               │
│ │Documents│ │Processed│ │Score Avg │ │Used     │               │
│ │ 1,247   │ │  189    │ │   87%    │ │ 847 GB  │               │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🤖 AI Activity Feed                                         │ │
│ │ • Processed: Privacy Policy 2024.pdf (95% compliance)      │ │
│ │ • Found gap: Cookie Policy missing GDPR consent language   │ │
│ │ • Suggestion: Update Terms of Service section 4.2          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Tasks Page (Compliance-Focused)

```
┌─────────────────────────────────────────────────────────────────┐
│ My Tasks & Assignments                                          │
│ Stay on top of your compliance work and deadlines              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [✅ My Tasks] [👥 Team Tasks] [🤖 AI Suggestions] [+ New Task]  │
│                                                                 │
│ Filter: [All Projects ▼] [All Status ▼] [Priority ▼]           │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Task                    │Priority│Due    │Project│Status   │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ 🔴 Review GDPR policy   │High   │Today  │GDPR  │In Progress│ │
│ │    Update consent forms │       │       │      │           │ │
│ │    🤖 AI found 3 gaps   │       │       │      │[Review]   │ │
│ │                         │       │       │      │           │ │
│ │ ⚠️ Upload SOC 2 docs    │Med    │Dec 15 │SOC 2 │Pending   │ │
│ │    Missing: Security    │       │       │      │           │ │
│ │    procedures manual    │       │       │      │[Upload]   │ │
│ │                         │       │       │      │           │ │
│ │ ✅ Complete HR audit    │Low    │Dec 20 │ISO   │Ready     │ │
│ │    checklist            │       │       │      │[Complete] │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Quick Stats                                                     │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│ │Total    │ │Due      │ │Completed│ │AI       │               │
│ │Tasks    │ │This Week│ │This Week│ │Suggested│               │
│ │   12    │ │   3     │ │    8    │ │   5     │               │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🤖 AI Recommendations                                       │ │
│ │ • Schedule quarterly GDPR review (suggested due: Mar 15)    │ │
│ │ • Create task: Update employee privacy training materials   │ │
│ │ • Priority: Review Cookie Policy compliance before Jan 1   │ │
│ │                                    [Create Tasks] [Dismiss] │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Reports Page (Executive Dashboard Focus)

```
┌─────────────────────────────────────────────────────────────────┐
│ Compliance Reports & Analytics                                  │
│ Monitor compliance health and generate executive reports        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 📊 Executive│ │ 📋 Generate │ │ 📅 Scheduled│               │
│ │ Dashboard   │ │ Custom      │ │ Reports     │               │
│ │ Real-time   │ │ Report      │ │ Automation  │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🛡️ Compliance Health Overview                              │ │
│ │                                                             │ │
│ │ Overall Score: 78% 🟡 (Target: 85%)                       │ │
│ │                                                             │ │
│ │ 🟢 GDPR: 95% ████████████████   🟡 ISO: 72% ██████████     │ │
│ │ 🟢 SOX:  88% █████████████      🔴 HIPAA: 45% ████        │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Quick Reports                                                   │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│ │Monthly  │ │Project  │ │Executive│ │Audit    │               │
│ │Summary  │ │Status   │ │Brief    │ │Ready    │               │
│ │Generate │ │Report   │ │PDF      │ │Export   │               │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
│                                                                 │
│ Recent Reports                      [View All] [Create Custom]  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Report Name         │Type    │Generated │Views│Status │Download│ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Q4 Compliance Brief │PDF     │Dec 10    │156  │✅    │[⬇️]   │ │
│ │ GDPR Monthly Status │Auto    │Dec 8     │89   │✅    │[⬇️]   │ │
│ │ SOC 2 Readiness     │Custom  │Dec 5     │234  │✅    │[⬇️]   │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 📈 Trends & Insights                                        │ │
│ │ • Compliance scores improved 15% this quarter               │ │
│ │ • HIPAA project needs immediate attention (30% vs 80% target)│ │
│ │ • Document processing time reduced by 40% with AI          │ │
│ │ • Next audit readiness: SOC 2 (95%), ISO (60% - at risk)  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ❓ Help Page (Getting Started Focus)

```
┌─────────────────────────────────────────────────────────────────┐
│ Help & Support                                                  │
│ Get up to speed quickly and find answers to your questions     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 🚀 Quick    │ │ 📖 User     │ │ 💬 Contact  │               │
│ │ Start Guide │ │ Manual      │ │ Support     │               │
│ │ Get started │ │ Full guide  │ │ Get help    │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🎯 New to DocCompliance Pro? Start here!                   │ │
│ │                                                             │ │
│ │ 1️⃣ Create your first compliance project                    │ │
│ │ 2️⃣ Upload documents and let AI analyze them               │ │
│ │ 3️⃣ Review AI suggestions and compliance gaps              │ │
│ │ 4️⃣ Generate your first compliance report                  │ │
│ │                                                             │ │
│ │              [▶️ Start Interactive Tour]                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Common Tasks                                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [📁] How do I create a new compliance project?             │ │
│ │ [📤] How do I upload and organize documents?               │ │
│ │ [🤖] How does AI document analysis work?                   │ │
│ │ [📊] How can I generate compliance reports?                │ │
│ │ [👥] How do I manage team permissions and roles?           │ │
│ │ [⚙️] Where are the advanced compliance features?           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Quick Resources                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [📹] Video Tutorials    [💬] Community Forum               │ │
│ │ [📧] Email Support      [📱] Mobile App Guide              │ │
│ │ [🔗] API Documentation  [📋] Template Library              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ⚡ Pro Tips                                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • Start with a small pilot project to learn the system     │ │
│ │ • Let AI agents run overnight for bulk document processing │ │
│ │ • Set up automated reports for regular compliance updates  │ │
│ │ • Use workspaces to organize different compliance areas    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Settings - Advanced Features Hub

### Settings Overview
```
┌─────────────────────────────────────────────────────────────────┐
│ Settings & Advanced Features                                    │
│ Configure system settings and access advanced compliance tools  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                      Settings Categories                     │ │
│ │                                                             │ │
│ │ 🛡️  Compliance Hub         📋  Policies & Templates       │ │
│ │     Full compliance center      Template management        │ │
│ │     [Advanced Tracking]         [Create Templates]         │ │
│ │                                                             │ │
│ │ 🤖  AI & Automation        🔧  System Administration       │ │
│ │     Agent management            User & system settings     │ │
│ │     [Configure Agents]          [User Management]          │ │
│ │                                                             │ │
│ │ 📈  Advanced Analytics     🔌  Integrations               │ │
│ │     Deep dive reporting         External connections       │ │
│ │     [Analytics Dashboard]       [API & Webhooks]          │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Quick Settings                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Account Settings    [⚙️]    Notification Preferences [🔔]  │ │
│ │ Security Settings   [🔒]    Workspace Management    [🏢]   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 🛡️ Settings > Compliance Hub (Full Original Features)
```
┌─────────────────────────────────────────────────────────────────┐
│ Settings > Compliance Hub                                       │
│ Advanced compliance tracking and management                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [🏠 Home] > [⚙️ Settings] > [🛡️ Compliance Hub]                │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 📊 Compliance│ │ 📋 Requirements│ │ 🔄 Workflows│               │
│ │ Overview     │ │ Management   │ │ & Approvals │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ Detailed Compliance Tracking (RAG Status)                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Framework           │Score│Trend│Last Review│Next│Assigned │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ 🟢 GDPR (EU)        │ 95% │ +3% │2024-12-01 │Mar │Jane S. │ │
│ │ 🟢 SOX (Financial)  │ 88% │ +1% │2024-11-15 │Feb │Mike J. │ │
│ │ 🟡 ISO 27001 (Info) │ 72% │ -2% │2024-12-10 │Jan │Alex K. │ │
│ │ 🟢 ISO 9001 (Quality)│ 91% │ +5% │2024-11-30 │Feb │Sarah L.│ │
│ │ 🔴 ISO 14001 (Env)  │ 45% │ -8% │2024-10-15 │⚠️  │Tom R.  │ │
│ │ 🟡 HIPAA (Health)   │ 67% │ +12%│2024-12-05 │Jan │Lisa M. │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Advanced Compliance Analytics                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [📊 Trend Analysis] [📈 Risk Assessment] [📋 Audit Trail]   │ │
│ │ [🎯 Gap Analysis]   [📅 Compliance Calendar] [📧 Alerts]    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 🤖 Settings > AI & Automation
```
┌─────────────────────────────────────────────────────────────────┐
│ Settings > AI & Automation                                      │
│ Configure AI agents and automation rules                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Active AI Agents                        [+ Deploy New Agent]    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Agent Name     │Status │Documents│Accuracy│Last Active      │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ 🤖 QA Primary   │🟢 Active│   247  │  92%  │2 min ago       │ │
│ │ 🛡️ Compliance   │🟢 Active│   156  │  89%  │5 min ago       │ │
│ │ 🔒 Security     │🟡 Limited│   89   │  94%  │1 hour ago      │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Agent Configuration                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Selected: [QA Primary ▼]                                   │ │
│ │                                                             │ │
│ │ Auto Processing: [ON]  │  Strict Mode: [OFF]              │ │
│ │ Confidence Level: [85%] ████████                          │ │
│ │ Processing Queue: 12 documents                             │ │
│ │                                                             │ │
│ │ Specialization: [GDPR] [SOX] [ISO 27001] [HIPAA]         │ │
│ │                                                             │ │
│ │ [⏸️ Pause] [🔄 Retrain] [📊 Performance] [⚙️ Advanced]    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Automation Rules                          [+ Create Rule]       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Rule Name              │Trigger    │Action         │Status  │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Auto-assign GDPR docs  │Doc upload │Assign project │🟢 On   │ │
│ │ Compliance threshold   │Score < 80%│Create task    │🟢 On   │ │
│ │ Weekly status report   │Every Mon  │Send email     │🟢 On   │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Recent AI Activity                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • Processed: Privacy Policy v2.1 (95% compliance)          │ │
│ │ • Gap identified: Terms of Service missing CCPA clause     │ │
│ │ • Task created: Update cookie consent mechanism            │ │
│ │ • Report generated: Weekly compliance summary              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 📋 Settings > Policies & Templates
```
┌─────────────────────────────────────────────────────────────────┐
│ Settings > Policies & Templates                                 │
│ Manage compliance templates, policies, and standards            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 📋 Templates│ │ 📜 Policies │ │ 🎯 Standards│               │
│ │ Library     │ │ Management  │ │ & Frameworks│               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ Template Library                        [+ Create] [Import]     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Template Name         │Framework│Version│Modified │Uses    │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ GDPR Compliance Kit   │GDPR     │v2.1   │Dec 5    │47 proj │ │
│ │ SOC 2 Type II Audit   │SOC 2    │v1.8   │Nov 20   │12 proj │ │
│ │ ISO 27001 Assessment  │ISO      │v3.0   │Oct 15   │23 proj │ │
│ │ HIPAA Security Rule   │HIPAA    │v1.5   │Sep 30   │8 proj  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Policy Management                       [+ New Policy]          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Policy Name           │Status   │Last Review│Next │Owner    │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Data Privacy Policy   │🟢 Active │Dec 1      │Mar 1│Legal   │ │
│ │ Information Security  │🟡 Review │Nov 15     │⚠️   │IT      │ │
│ │ Employee Handbook     │🟢 Active │Oct 30     │Jan 30│HR     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Compliance Standards Configuration                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Framework    │Enabled│Threshold│Auto-Check│Notifications    │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ GDPR         │  ✅   │  85%    │    ✅     │ Email + Slack  │ │
│ │ SOX          │  ✅   │  90%    │    ✅     │ Email only     │ │
│ │ ISO 27001    │  ✅   │  80%    │    ⬜     │ Dashboard only │ │
│ │ HIPAA        │  ✅   │  95%    │    ✅     │ All channels   │ │
│ │ PCI DSS      │  ⬜   │  --     │    ⬜     │ --             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 🔧 Settings > System Administration
```
┌─────────────────────────────────────────────────────────────────┐
│ Settings > System Administration                                │
│ Manage users, system logs, and administrative settings         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│ │ 👥 User     │ │ 📋 System   │ │ 🔌 Integrations │               │
│ │ Management  │ │ Logs        │ │ & API Keys   │               │
│ └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
│ User Management                         [+ Invite User]         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Name           │Email          │Role        │Status │Last   │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ John Doe       │john@...       │Admin       │🟢     │2m ago │ │
│ │ Jane Smith     │jane@...       │Compliance  │🟢     │1h ago │ │
│ │ Mike Johnson   │mike@...       │Viewer      │🟡     │2d ago │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ System Health                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 🟢 All Systems Operational                                  │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │ │
│ │ │Database     │ │AI Agents    │ │File Storage │           │ │
│ │ │🟢 Healthy   │ │🟢 Running   │ │🟢 Available │           │ │
│ │ │99.9% uptime │ │3 active     │ │847 GB used  │           │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Recent System Activity                  [📋 View Full Logs]     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • User login: jane@company.com (2024-12-13 10:30)          │ │
│ │ • Document processed: Privacy_Policy_v2.pdf (95% score)    │ │
│ │ • Backup completed: Daily backup successful                │ │
│ │ • Integration sync: Slack notifications updated            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Security & Backup Settings                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Two-Factor Auth: [✅ Enabled]    Auto Backup: [✅ Daily]    │ │
│ │ Session Timeout: [30 minutes]    Retention: [7 years]      │ │
│ │ Password Policy: [✅ Enforced]   Encryption: [✅ AES-256]  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Unified User Flow Patterns

### Primary User Journey (Simplified)
```
1. 🏠 Welcome → Quick overview + immediate actions
   ├─ Create Project → Guided project setup
   ├─ Upload Documents → AI-powered processing
   └─ Check Tasks → Action items from AI analysis

2. 📁 Projects → Main workflow hub
   ├─ Project Details → Documents, compliance, progress
   ├─ Team Collaboration → Task assignment, comments
   └─ AI Insights → Automated compliance suggestions

3. 📊 Reports → Executive dashboard
   ├─ Compliance Health → RAG status overview
   ├─ Generate Reports → Automated + custom
   └─ Trend Analysis → Performance over time

4. ⚙️ Settings → Advanced features (for power users)
   ├─ 🛡️ Compliance Hub → Full compliance center
   ├─ 🤖 AI & Automation → Agent configuration
   ├─ 📋 Policies & Templates → Template management
   └─ 🔧 System Admin → User and system management
```

### Navigation Philosophy
```
Simple → Advanced
┌─────────────────┐    ┌─────────────────┐
│   Main UI       │    │   Settings      │
│   (ProjectFlow  │    │   (Advanced     │
│    Simplicity)  │───▶│    Features)    │
│                 │    │                 │
│ • Projects      │    │ • Compliance Hub│
│ • Documents     │    │ • AI Management │
│ • Tasks         │    │ • Templates     │
│ • Reports       │    │ • System Admin  │
│ • Help          │    │ • Analytics     │
└─────────────────┘    └─────────────────┘

🎯 90% of users stay in main UI
⚙️ 10% power users access Settings for advanced features
```

---

## 📱 Responsive Design Patterns

### Desktop (1280px+)
```
┌─────┬─────────────────────────────────────────┐
│     │           Top Nav + Search              │
│ S   ├─────────────────────────────────────────┤
│ i   │                                         │
│ d   │            Main Content                 │
│ e   │              Area                       │
│ b   │                                         │
│ a   │                                         │
│ r   │                                         │
│     │                                         │
└─────┴─────────────────────────────────────────┘
```

### Tablet (768px-1279px)
```
┌─────────────────────────────────────────┐
│        Top Nav + Hamburger              │
├─────────────────────────────────────────┤
│                                         │
│            Main Content                 │
│      (Sidebar collapses to overlay)     │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

### Mobile (≤767px)
```
┌─────────────────────────────────────────┐
│  [☰] DocCompliance Pro      [👤] [🔔]  │
├─────────────────────────────────────────┤
│                                         │
│            Main Content                 │
│           (Single column)               │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│ [🏠] [📁] [📄] [✅] [📊]               │
└─────────────────────────────────────────┘
Optional: Bottom tab navigation for mobile
```

---

## 🎨 Unified Design System

### Color Philosophy
```
Primary Workflow (ProjectFlow inspired):
🔵 Blue (#2563eb) - Primary actions, navigation
🟢 Green (#16a34a) - Success, completed, healthy
🟡 Yellow (#f59e0b) - Warning, attention needed
🔴 Red (#ef4444) - Error, critical, overdue

Advanced Features (Professional):
🟣 Purple (#8b5cf6) - AI/automation features
🟠 Orange (#f97316) - Analytics and insights
⚫ Gray scale - Background, text, borders
```

### Component Consistency
```
Card Pattern (Used throughout):
┌─────────────────────────────────┐
│ [Icon] Title            [Badge] │
│ Description or content...       │
│ ─────────────────────────────── │
│ Footer info    [Action Button]  │
└─────────────────────────────────┘

Data Table Pattern:
┌─────────────────────────────────────────────┐
│ Table Title              [Filter] [Actions] │
├─────────────────────────────────────────────┤
│ Col 1   │ Col 2   │ Status │ Progress │ ... │
│ Data... │ Value..│   🟢   │ ████     │ [⚙️] │
└─────────────────────────────────────────────┘
```

---

## 🚀 Key Benefits of Unified Design

### User Experience
- **Familiar Flow**: ProjectFlow simplicity for main tasks
- **Progressive Disclosure**: Advanced features hidden in Settings
- **Consistent Patterns**: Same UI components throughout
- **Context-Aware**: Smart defaults based on user behavior

### Functional Benefits
- **No Lost Features**: All original DocCompliance Pro functionality preserved
- **Easier Onboarding**: Simple project-based workflow to start
- **Scalable Complexity**: Users can grow into advanced features
- **Enterprise Ready**: Full compliance management under the hood

### Technical Architecture
- **Single Codebase**: Shared components between simple and advanced views
- **Progressive Enhancement**: Core features work, advanced features enhance
- **Role-Based Access**: UI adapts to user permissions and experience level
- **API Consistency**: Same backend serves both simple and complex interfaces

---

This unified wireframe creates the best of both worlds: the approachable, project-focused workflow of ProjectFlow for everyday users, while maintaining all the sophisticated compliance management capabilities of DocCompliance Pro in an organized, discoverable way through the Settings hub. Users can start simple and grow into the full power of the system as their needs mature.
│
