import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { TrendingUp, Activity, Target, BarChart3 } from "lucide-react";

export function ConsistencyTab() {
  return (
    <div className="space-y-8">
      {/* Simple Header */}
      <Card className="border-l-8 border-l-[#238196] shadow-xl">
        <CardHeader className="text-center py-6">
          <div className="mx-auto w-16 h-16 bg-[#238196] rounded-full flex items-center justify-center mb-4">
            <TrendingUp className="h-8 w-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-[#238196] mb-2">📈 Consistency Results</CardTitle>
          <p className="text-lg text-muted-foreground">
            How stable your model performs over time
          </p>
        </CardHeader>
      </Card>

      {/* Current Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="shadow-xl bg-gradient-to-br from-green-50 to-white border-2 border-green-200">
          <CardContent className="p-8 text-center">
            <Activity className="h-16 w-16 text-green-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-green-700 mb-2">Model Drift</h3>
            <p className="text-5xl font-bold text-green-600 my-4">2.3%</p>
            <p className="text-lg text-green-600">✅ Really Good!</p>
            <p className="text-sm text-green-600 mt-2">Below 5% is excellent</p>
          </CardContent>
        </Card>
        
        <Card className="shadow-xl bg-gradient-to-br from-blue-50 to-white border-2 border-blue-200">
          <CardContent className="p-8 text-center">
            <Target className="h-16 w-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-blue-700 mb-2">Stability</h3>
            <p className="text-5xl font-bold text-blue-600 my-4">87%</p>
            <p className="text-lg text-blue-600">✅ Good!</p>
            <p className="text-sm text-blue-600 mt-2">Above 85% is solid</p>
          </CardContent>
        </Card>
        
        <Card className="shadow-xl bg-gradient-to-br from-purple-50 to-white border-2 border-purple-200">
          <CardContent className="p-8 text-center">
            <BarChart3 className="h-16 w-16 text-purple-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-purple-700 mb-2">Quality Score</h3>
            <p className="text-5xl font-bold text-purple-600 my-4">94%</p>
            <p className="text-lg text-purple-600">🌟 Excellent!</p>
            <p className="text-sm text-purple-600 mt-2">Top 10% performance</p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Over Time */}
      <Card className="shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl text-[#238196]">📅 Performance History</CardTitle>
          <p className="text-lg text-muted-foreground">How your model has been doing each month</p>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-[#238196]/10 to-white rounded-xl border-2 border-[#238196]/20">
              <div className="w-16 h-16 bg-[#238196] rounded-full flex items-center justify-center">
                <span className="text-white text-xl font-bold">Jan</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium text-[#238196]">🚀 Started Great</p>
                <p className="text-lg text-muted-foreground">Model launched with 95% accuracy</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-[#238196]">95%</p>
                <p className="text-sm text-[#238196]">Excellent start</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-[#085280]/10 to-white rounded-xl border-2 border-[#085280]/20">
              <div className="w-16 h-16 bg-[#085280] rounded-full flex items-center justify-center">
                <span className="text-white text-xl font-bold">Feb</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium text-[#085280]">📊 Stayed Stable</p>
                <p className="text-lg text-muted-foreground">Small dip but still performing well</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-[#085280]">91%</p>
                <p className="text-sm text-[#085280]">Still good</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-[#C55422]/10 to-white rounded-xl border-2 border-[#C55422]/20">
              <div className="w-16 h-16 bg-[#C55422] rounded-full flex items-center justify-center">
                <span className="text-white text-xl font-bold">Mar</span>
              </div>
              <div className="flex-1">
                <p className="text-xl font-medium text-[#C55422]">⚠️ Small Drop</p>
                <p className="text-lg text-muted-foreground">Performance dipped - needs monitoring</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-[#C55422]">87%</p>
                <p className="text-sm text-[#C55422]">Watch closely</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* What This Means */}
      <Card className="shadow-xl bg-gradient-to-r from-blue-50 to-white">
        <CardHeader>
          <CardTitle className="text-2xl text-[#238196]">🤔 What This Means</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-6 bg-white rounded-xl shadow border-l-4 border-l-green-500">
              <p className="text-xl font-medium text-green-700 mb-2">✅ Good News</p>
              <p className="text-lg">Your model is pretty stable and hasn't drifted much. It's working consistently!</p>
            </div>
            <div className="p-6 bg-white rounded-xl shadow border-l-4 border-l-yellow-500">
              <p className="text-xl font-medium text-yellow-700 mb-2">⚠️ Watch Out</p>
              <p className="text-lg">There's a small downward trend. Keep an eye on it and be ready to retrain if it gets worse.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}