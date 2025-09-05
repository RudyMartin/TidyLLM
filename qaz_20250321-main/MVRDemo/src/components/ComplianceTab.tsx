import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { FileText } from "lucide-react";

export function ComplianceTab() {
  return (
    <div className="space-y-8">
      {/* Simple Header */}
      <Card className="border-l-8 border-l-[#085280] shadow-xl">
        <CardHeader className="text-center py-6">
          <div className="mx-auto w-16 h-16 bg-[#085280] rounded-full flex items-center justify-center mb-4">
            <FileText className="h-8 w-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-[#085280] mb-2">📋 Compliance Results</CardTitle>
          <p className="text-lg text-muted-foreground">
            How well your model follows the rules
          </p>
        </CardHeader>
      </Card>

      {/* Big Score Display */}
      <Card className="shadow-xl border-l-8 border-l-[#C55422]">
        <CardContent className="text-center py-8">
          <div className="space-y-6">
            {/* Big Score */}
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-[#085280] to-[#238196] rounded-full flex items-center justify-center">
              <span className="text-5xl font-bold text-white">78%</span>
            </div>
            
            <div className="space-y-4">
              <Badge variant="outline" className="text-2xl px-6 py-3 text-[#C55422] border-[#C55422]">
                ⚠️ Needs Work
              </Badge>
              
              <p className="text-xl text-muted-foreground">
                Your model follows most rules but has some issues to fix
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Breakdown */}
      <Card className="shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl text-[#085280]">📊 What We Found</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-6">
            <div className="p-6 bg-red-50 rounded-xl border-2 border-red-200">
              <div className="text-center">
                <p className="text-3xl font-bold text-red-700 mb-2">70%</p>
                <p className="text-xl font-medium text-red-700">❌ Risk Controls</p>
                <p className="text-sm text-red-600 mt-2">Need better monitoring</p>
              </div>
            </div>
            <div className="p-6 bg-yellow-50 rounded-xl border-2 border-yellow-200">
              <div className="text-center">
                <p className="text-3xl font-bold text-yellow-700 mb-2">75%</p>
                <p className="text-xl font-medium text-yellow-700">⚠️ Data Quality</p>
                <p className="text-sm text-yellow-600 mt-2">Some gaps found</p>
              </div>
            </div>
            <div className="p-6 bg-blue-50 rounded-xl border-2 border-blue-200">
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-700 mb-2">82%</p>
                <p className="text-xl font-medium text-blue-700">📋 Rule Following</p>
                <p className="text-sm text-blue-600 mt-2">Pretty good!</p>
              </div>
            </div>
            <div className="p-6 bg-green-50 rounded-xl border-2 border-green-200">
              <div className="text-center">
                <p className="text-3xl font-bold text-green-700 mb-2">84%</p>
                <p className="text-xl font-medium text-green-700">✅ Documentation</p>
                <p className="text-sm text-green-600 mt-2">Well documented</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Plan */}
      <Card className="shadow-xl bg-gradient-to-r from-[#C55422]/10 to-white">
        <CardHeader>
          <CardTitle className="text-2xl text-[#C55422]">🎯 Fix These Things First</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4 p-6 bg-white rounded-lg shadow border-l-4 border-l-red-500">
              <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-xl">1</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium">Add Better Risk Monitoring</p>
                <p className="text-lg text-muted-foreground">Set up alerts when things go wrong</p>
              </div>
              <Badge className="bg-red-100 text-red-700 border-2 border-red-200 text-base px-4 py-2">
                HIGH PRIORITY
              </Badge>
            </div>
            
            <div className="flex items-center gap-4 p-6 bg-white rounded-lg shadow border-l-4 border-l-yellow-500">
              <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-xl">2</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium">Fix Data Quality Issues</p>
                <p className="text-lg text-muted-foreground">Clean up missing and wrong data</p>
              </div>
              <Badge className="bg-yellow-100 text-yellow-700 border-2 border-yellow-200 text-base px-4 py-2">
                MEDIUM PRIORITY
              </Badge>
            </div>
            
            <div className="flex items-center gap-4 p-6 bg-white rounded-lg shadow border-l-4 border-l-blue-500">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-xl">3</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium">Update Some Rules</p>
                <p className="text-lg text-muted-foreground">Make sure all new regulations are followed</p>
              </div>
              <Badge className="bg-blue-100 text-blue-700 border-2 border-blue-200 text-base px-4 py-2">
                LOW PRIORITY
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}