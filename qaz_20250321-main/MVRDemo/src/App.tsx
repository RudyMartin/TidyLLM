import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Upload, CheckCircle, FileText, TrendingUp, Users, BarChart3 } from "lucide-react";

export default function App() {
  const [step, setStep] = useState(1);
  const [hasFile, setHasFile] = useState(false);
  const [selectedReport, setSelectedReport] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileUpload = () => {
    setHasFile(true);
    setStep(2);
  };

  const handleReportSelect = (reportType: string) => {
    setSelectedReport(reportType);
    setStep(3);
    setIsAnalyzing(true);
    // Simulate analysis
    setTimeout(() => {
      setIsAnalyzing(false);
    }, 3000);
  };

  const resetApp = () => {
    setStep(1);
    setHasFile(false);
    setSelectedReport("");
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-4xl mx-auto p-6">
        
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-[#085280] to-[#238196] bg-clip-text text-transparent">
            MVR Review
          </h1>
          <p className="text-muted-foreground text-xl">Model Validation & QC Assessment</p>
          
          {/* Progress Steps */}
          <div className="flex items-center justify-center gap-4 mt-8">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${step >= 1 ? 'bg-[#085280] text-white' : 'bg-gray-200 text-gray-500'}`}>
              <span className="font-bold">1</span>
              <span>Upload Files</span>
            </div>
            <div className="w-8 h-1 bg-gray-300 rounded"></div>
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${step >= 2 ? 'bg-[#238196] text-white' : 'bg-gray-200 text-gray-500'}`}>
              <span className="font-bold">2</span>
              <span>Pick Report</span>
            </div>
            <div className="w-8 h-1 bg-gray-300 rounded"></div>
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${step >= 3 ? 'bg-[#C55422] text-white' : 'bg-gray-200 text-gray-500'}`}>
              <span className="font-bold">3</span>
              <span>Review Results</span>
            </div>
          </div>
        </div>

        {/* Step 1: Upload Files */}
        {step === 1 && (
          <Card className="shadow-xl border-2 border-[#085280]/20">
            <CardHeader className="text-center py-8">
              <CardTitle className="text-3xl text-[#085280] mb-4">📁 Step 1: Upload Your Model Files</CardTitle>
              <p className="text-xl text-muted-foreground">
                Choose the files you want to review
              </p>
            </CardHeader>
            <CardContent className="p-8">
              <div className="text-center space-y-6">
                <div className="w-24 h-24 mx-auto bg-gray-100 rounded-full flex items-center justify-center border-4 border-dashed border-gray-300">
                  <Upload className="h-12 w-12 text-gray-400" />
                </div>
                <div className="space-y-4">
                  <p className="text-lg text-muted-foreground">Supported files: .sql, .py, .json, .csv</p>
                  <Button 
                    size="lg" 
                    onClick={handleFileUpload}
                    className="h-16 px-12 text-xl bg-gradient-to-r from-[#085280] to-[#238196] hover:from-[#085280]/90 hover:to-[#238196]/90"
                  >
                    <Upload className="h-6 w-6 mr-3" />
                    📁 Choose Files
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 2: Pick Report Type */}
        {step === 2 && (
          <Card className="shadow-xl border-2 border-[#238196]/20">
            <CardHeader className="text-center py-8">
              <CardTitle className="text-3xl text-[#238196] mb-4">📋 Step 2: Pick Your Report Type</CardTitle>
              <p className="text-xl text-muted-foreground">
                What kind of analysis do you want?
              </p>
              {hasFile && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg border-2 border-green-200">
                  <div className="flex items-center justify-center gap-3">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                    <span className="text-lg font-medium text-green-700">✅ MPM_Model.sql uploaded (14 KB)</span>
                  </div>
                </div>
              )}
            </CardHeader>
            <CardContent className="p-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <Card 
                  className="border-2 border-[#085280]/20 hover:border-[#085280] hover:shadow-xl transition-all cursor-pointer"
                  onClick={() => handleReportSelect("compliance")}
                >
                  <CardContent className="p-8 text-center">
                    <div className="w-16 h-16 mx-auto bg-[#085280] rounded-full flex items-center justify-center mb-4">
                      <FileText className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-[#085280] mb-2">Compliance Report</h3>
                    <p className="text-muted-foreground mb-4">Check if your model follows all the rules and regulations</p>
                    <Badge className="bg-[#085280]/10 text-[#085280] border-[#085280]/20">
                      Most Popular
                    </Badge>
                  </CardContent>
                </Card>

                <Card 
                  className="border-2 border-[#238196]/20 hover:border-[#238196] hover:shadow-xl transition-all cursor-pointer"
                  onClick={() => handleReportSelect("consistency")}
                >
                  <CardContent className="p-8 text-center">
                    <div className="w-16 h-16 mx-auto bg-[#238196] rounded-full flex items-center justify-center mb-4">
                      <TrendingUp className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-[#238196] mb-2">Consistency Report</h3>
                    <p className="text-muted-foreground mb-4">See how stable your model performs over time</p>
                    <Badge className="bg-[#238196]/10 text-[#238196] border-[#238196]/20">
                      Recommended
                    </Badge>
                  </CardContent>
                </Card>

                <Card 
                  className="border-2 border-[#C55422]/20 hover:border-[#C55422] hover:shadow-xl transition-all cursor-pointer"
                  onClick={() => handleReportSelect("challenge")}
                >
                  <CardContent className="p-8 text-center">
                    <div className="w-16 h-16 mx-auto bg-[#C55422] rounded-full flex items-center justify-center mb-4">
                      <Users className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-[#C55422] mb-2">Challenge Report</h3>
                    <p className="text-muted-foreground mb-4">Test your model with tough scenarios and peer reviews</p>
                    <Badge className="bg-[#C55422]/10 text-[#C55422] border-[#C55422]/20">
                      Advanced
                    </Badge>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 3: Results */}
        {step === 3 && (
          <div className="space-y-6">
            
            {/* Analysis Status */}
            {isAnalyzing && (
              <Card className="shadow-xl border-2 border-[#C55422]/20">
                <CardContent className="p-8 text-center">
                  <div className="w-16 h-16 mx-auto bg-[#C55422] rounded-full flex items-center justify-center mb-4 animate-pulse">
                    <BarChart3 className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-[#C55422] mb-2">🔄 Analyzing Your Model...</h3>
                  <p className="text-lg text-muted-foreground">
                    Running {selectedReport} analysis on MPM_Model.sql
                  </p>
                  <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-gradient-to-r from-[#085280] to-[#238196] h-2 rounded-full animate-pulse" style={{width: '75%'}}></div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Results */}
            {!isAnalyzing && (
              <>
                <Card className="shadow-xl border-2 border-green-200">
                  <CardHeader className="text-center py-6">
                    <CardTitle className="text-3xl text-green-700 mb-2">✅ Analysis Complete!</CardTitle>
                    <p className="text-xl text-muted-foreground capitalize">
                      {selectedReport} Report for MPM_Model.sql
                    </p>
                  </CardHeader>
                </Card>

                {/* Compliance Results */}
                {selectedReport === "compliance" && (
                  <div className="space-y-6">
                    <Card className="shadow-xl">
                      <CardContent className="text-center py-8">
                        <div className="w-32 h-32 mx-auto bg-gradient-to-br from-[#085280] to-[#238196] rounded-full flex items-center justify-center mb-6">
                          <span className="text-5xl font-bold text-white">78%</span>
                        </div>
                        <Badge variant="outline" className="text-2xl px-6 py-3 text-[#C55422] border-[#C55422] mb-4">
                          ⚠️ Needs Work
                        </Badge>
                        <p className="text-xl text-muted-foreground">
                          Your model follows most rules but has some issues to fix
                        </p>
                      </CardContent>
                    </Card>

                    <Card className="shadow-xl">
                      <CardHeader>
                        <CardTitle className="text-2xl text-[#085280]">📊 What We Found</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-red-50 rounded-lg border-2 border-red-200 text-center">
                            <p className="text-2xl font-bold text-red-700">70%</p>
                            <p className="text-lg font-medium text-red-700">❌ Risk Controls</p>
                          </div>
                          <div className="p-4 bg-yellow-50 rounded-lg border-2 border-yellow-200 text-center">
                            <p className="text-2xl font-bold text-yellow-700">75%</p>
                            <p className="text-lg font-medium text-yellow-700">⚠️ Data Quality</p>
                          </div>
                          <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-200 text-center">
                            <p className="text-2xl font-bold text-blue-700">82%</p>
                            <p className="text-lg font-medium text-blue-700">📋 Rules</p>
                          </div>
                          <div className="p-4 bg-green-50 rounded-lg border-2 border-green-200 text-center">
                            <p className="text-2xl font-bold text-green-700">84%</p>
                            <p className="text-lg font-medium text-green-700">✅ Documentation</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}

                {/* Consistency Results */}
                {selectedReport === "consistency" && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-3 gap-6">
                      <Card className="shadow-xl bg-gradient-to-br from-green-50 to-white border-2 border-green-200">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-green-700 mb-2">Model Drift</h3>
                          <p className="text-4xl font-bold text-green-600">2.3%</p>
                          <p className="text-sm text-green-600">✅ Really Good!</p>
                        </CardContent>
                      </Card>
                      
                      <Card className="shadow-xl bg-gradient-to-br from-blue-50 to-white border-2 border-blue-200">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-blue-700 mb-2">Stability</h3>
                          <p className="text-4xl font-bold text-blue-600">87%</p>
                          <p className="text-sm text-blue-600">✅ Good!</p>
                        </CardContent>
                      </Card>
                      
                      <Card className="shadow-xl bg-gradient-to-br from-purple-50 to-white border-2 border-purple-200">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-purple-700 mb-2">Quality</h3>
                          <p className="text-4xl font-bold text-purple-600">94%</p>
                          <p className="text-sm text-purple-600">🌟 Excellent!</p>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}

                {/* Challenge Results */}
                {selectedReport === "challenge" && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-3 gap-6">
                      <Card className="shadow-xl bg-gradient-to-br from-[#C55422]/10 to-white border-2 border-[#C55422]/20">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-[#C55422] mb-2">Peer Reviews</h3>
                          <p className="text-4xl font-bold text-[#C55422]">12</p>
                          <p className="text-sm text-muted-foreground">experts reviewing</p>
                        </CardContent>
                      </Card>
                      
                      <Card className="shadow-xl bg-gradient-to-br from-[#238196]/10 to-white border-2 border-[#238196]/20">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-[#238196] mb-2">Stress Tests</h3>
                          <p className="text-4xl font-bold text-[#238196]">8</p>
                          <p className="text-sm text-muted-foreground">scenarios passed</p>
                        </CardContent>
                      </Card>
                      
                      <Card className="shadow-xl bg-gradient-to-br from-[#085280]/10 to-white border-2 border-[#085280]/20">
                        <CardContent className="p-6 text-center">
                          <h3 className="text-lg font-bold text-[#085280] mb-2">Edge Cases</h3>
                          <p className="text-4xl font-bold text-[#085280]">15</p>
                          <p className="text-sm text-muted-foreground">handled correctly</p>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <Card className="shadow-xl">
                  <CardContent className="p-6">
                    <div className="flex gap-4 justify-center">
                      <Button 
                        size="lg" 
                        onClick={resetApp}
                        variant="outline"
                        className="h-14 px-8 text-lg border-2 border-[#085280] text-[#085280] hover:bg-[#085280]/10"
                      >
                        🔄 Start Over
                      </Button>
                      <Button 
                        size="lg" 
                        className="h-14 px-8 text-lg bg-gradient-to-r from-[#085280] to-[#238196] hover:from-[#085280]/90 hover:to-[#238196]/90"
                      >
                        📄 Download Report
                      </Button>
                      <Button 
                        size="lg" 
                        onClick={() => setStep(2)}
                        variant="outline"
                        className="h-14 px-8 text-lg border-2 border-[#238196] text-[#238196] hover:bg-[#238196]/10"
                      >
                        📊 Try Different Report
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}