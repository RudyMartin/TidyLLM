Complete Document Queue Recognition Workflow Analysis

I've analyzed the current document lifecycle and created a comprehensive overview of how new documents flow through the QA system to be recognized by users.

📋 Current Document Recognition Process:

Entry Points:
Single/Batch uploads through DocumentUpload/BatchUpload components
Mock data generation in demo mode  
Future API/SharePoint integration capabilities

Recognition Workflow (5 Stages):

📤 Document Upload (`status: 'draft'`)
Immediate visibility in "New" quick link
Auto-generated audit entry
Added to allDocuments state

⏰ Awaiting Assignment (`status: 'pending'`)  
Appears in "Pending" quick link
Visible in Analysis > Pending Reviews
Ready for reviewer assignment

👤 Review Assigned (`status: 'in-progress'`)
Shows in user's "MyTasks" quick link when `assignedReviewer` matches email
Becomes actionable for assigned QA user
Status tracking begins

✏️ Active Review (`status: 'partial'`)
Active compliance scoring in progress
Real-time progress tracking
User-specific task management

✅ Review Complete (`status: 'complete'`)
Finalized compliance scores
Full audit trail available
Results dashboard integration

🎯 Key QA User Recognition Mechanisms:

Compact Quick Links (Header):
New (📄) - Today's uploads via `uploadedAt` filtering
MyTasks (👤) - User-assigned documents via `assignedReviewer` 
Pending (⏰) - Awaiting review via status filtering
Failed (⚠️) - Error/failed documents needing attention
Blockers (🛡️) - Overdue/blocked items via `lastActivity`

Navigation Recognition:
Upload > Document Selection - All uploaded files
Analysis > Pending Reviews - Review queue
Results > Dashboard - Complete overview with filtering

Automatic Triggers:
Time-based recognition (today's uploads)
Assignment-based (user email matching)
Status-based (workflow progression)  
Priority-based (urgent/high priority)
Activity-based (overdue detection)

🔄 Created DocumentQueueWorkflow Component:
Visual workflow pipeline showing all 5 stages
Real-time metrics and counts per stage
Interactive stage exploration with document samples
Recognition point explanations for each stage
Progress tracking and completion percentages

The system provides multiple recognition touchpoints ensuring QA users can efficiently identify and act on documents requiring their attention at every stage of the compliance review process! 
