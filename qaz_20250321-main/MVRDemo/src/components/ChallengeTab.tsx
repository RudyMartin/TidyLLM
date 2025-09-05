import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Users, Brain, AlertTriangle, Target } from "lucide-react";

export function ChallengeTab() {
  return (
    <div className="space-y-8">
      {/* Simple Header */}
      <Card className="border-l-8 border-l-[#C55422] shadow-xl">
        <CardHeader className="text-center py-6">
          <div className="mx-auto w-16 h-16 bg-[#C55422] rounded-full flex items-center justify-center mb-4">
            <Users className="h-8 w-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-[#C55422] mb-2">🎯 Challenge Results</CardTitle>
          <p className="text-lg text-muted-foreground">
            How your model handles tough tests and peer reviews
          </p>
        </CardHeader>
      </Card>

      {/* Current Activity Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="shadow-xl bg-gradient-to-br from-[#C55422]/10 to-white border-2 border-[#C55422]/20">
          <CardContent className="p-8 text-center">
            <Brain className="h-16 w-16 text-[#C55422] mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-[#C55422] mb-2">Peer Reviews</h3>
            <p className="text-5xl font-bold text-[#C55422] my-4">12</p>
            <p className="text-lg text-muted-foreground mb-2">experts reviewing</p>
            <Badge className="bg-[#C55422]/10 text-[#C55422] border-2 border-[#C55422]/20">
              3 ACTIVE NOW
            </Badge>
          </CardContent>
        </Card>
        
        <Card className="shadow-xl bg-gradient-to-br from-[#238196]/10 to-white border-2 border-[#238196]/20">
          <CardContent className="p-8 text-center">
            <AlertTriangle className="h-16 w-16 text-[#238196] mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-[#238196] mb-2">Stress Tests</h3>
            <p className="text-5xl font-bold text-[#238196] my-4">8</p>
            <p className="text-lg text-muted-foreground mb-2">tough scenarios</p>
            <Badge className="bg-yellow-100 text-yellow-700 border-2 border-yellow-200">
              2 NEED WORK
            </Badge>
          </CardContent>
        </Card>
        
        <Card className="shadow-xl bg-gradient-to-br from-[#085280]/10 to-white border-2 border-[#085280]/20">
          <CardContent className="p-8 text-center">
            <Target className="h-16 w-16 text-[#085280] mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-[#085280] mb-2">Edge Cases</h3>
            <p className="text-5xl font-bold text-[#085280] my-4">15</p>
            <p className="text-lg text-muted-foreground mb-2">weird situations</p>
            <Badge className="bg-green-100 text-green-700 border-2 border-green-200">
              MOSTLY GOOD
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Recent Test Results */}
      <Card className="shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl text-[#C55422]">🏆 Latest Test Results</CardTitle>
          <p className="text-lg text-muted-foreground">How your model did on recent challenges</p>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-green-50 to-white rounded-xl border-2 border-green-200">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center">
                <span className="text-3xl">✅</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-bold text-green-700">Edge Case Test: PASSED</p>
                <p className="text-lg text-green-600">Handled 95% of weird situations correctly</p>
                <p className="text-sm text-green-600 mt-1">Completed 2 hours ago</p>
              </div>
              <Badge className="bg-green-100 text-green-700 border-2 border-green-200 text-lg px-4 py-2">
                EXCELLENT
              </Badge>
            </div>
            
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-yellow-50 to-white rounded-xl border-2 border-yellow-200">
              <div className="w-16 h-16 bg-yellow-500 rounded-full flex items-center justify-center">
                <span className="text-3xl">⚠️</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-bold text-yellow-700">Stress Test: ISSUES FOUND</p>
                <p className="text-lg text-yellow-600">Got slow when handling lots of data at once</p>
                <p className="text-sm text-yellow-600 mt-1">Completed yesterday</p>
              </div>
              <Badge className="bg-yellow-100 text-yellow-700 border-2 border-yellow-200 text-lg px-4 py-2">
                NEEDS WORK
              </Badge>
            </div>
            
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-[#238196]/10 to-white rounded-xl border-2 border-[#238196]/20">
              <div className="w-16 h-16 bg-[#238196] rounded-full flex items-center justify-center">
                <span className="text-3xl">👥</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-bold text-[#238196]">Peer Review: IN PROGRESS</p>
                <p className="text-lg text-[#238196]">Dr. Smith, Dr. Jones, and Dr. Lee are reviewing</p>
                <p className="text-sm text-[#238196] mt-1">Started this morning</p>
              </div>
              <Badge className="bg-[#238196]/10 text-[#238196] border-2 border-[#238196]/20 text-lg px-4 py-2">
                REVIEWING
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="shadow-xl bg-gradient-to-r from-[#C55422]/5 to-white">
        <CardHeader>
          <CardTitle className="text-2xl text-[#C55422]">🚀 Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Button 
              size="lg" 
              className="h-16 text-lg bg-gradient-to-r from-[#C55422] to-[#238196] hover:from-[#C55422]/90 hover:to-[#238196]/90"
            >
              👥 Request More Reviews
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="h-16 text-lg border-2 border-[#238196] text-[#238196] hover:bg-[#238196]/10"
            >
              📋 View Full Report
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}