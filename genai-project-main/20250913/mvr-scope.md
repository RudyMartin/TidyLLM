 Flow Overview

Flow Name: mvr_scope

Purpose: This is a QA check. It Reviews MVR document section by section using prompts for each of the nine sections. For all the items in the sections checks for vst_details and if htat has as child field in_scope: true then added that to total_in_scope, also extracts fields from VST document and determines section by section if child fil is_applicable:true then adds that to total applicable. Determines total_in_scope / total_applicable_scope and shows overlap and outer sets of sections using titles and section numbers.

Trigger: Someone uploads a file

📥 Input/Output Specification

Inputs:

MVR document, VST document, Critieria json, Instructions to start in master_template.md and section_{x}_template.md

Outputs:

a json with commentary and results per instructions in markdown

⚙️ Processing Steps

Ingest an Extract VST content into vst_json
Merge that with criteria_json
Ingest MVR and Run section by section markdown templates
Add Text of Section and AIAnalysis to the merged json
SAVE merged json to output folder

📋 Criteria and Rules

Quality Rules:

VST and MVR should have TOC for guidance of sections - alternative is to match same using titles and placement of sections

Scoring Method: 1-10 scale

HIgh score = complete MVR with all is_applicable sections matching in_sope

🔧 Technical Requirements

Must meet compliance standards
Must handle high volume
Additional Notes:

Read master_template.md and then For each section there will be a template. Loop through the 9 sections and extract text and response in the merger_json file.
