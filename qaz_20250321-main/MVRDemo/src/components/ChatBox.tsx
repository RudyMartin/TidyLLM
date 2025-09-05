import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";
import { MessageCircle, Send, Bot, User, Minimize2, Maximize2 } from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  type?: 'command' | 'analysis' | 'general';
}

export function ChatBox() {
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputMessage, setInputMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hi! I'm here to help you check your models. Ask me anything or try the quick buttons below! 😊",
      sender: 'assistant',
      timestamp: new Date(Date.now() - 300000),
      type: 'general'
    },
    {
      id: "2",
      content: "How is my model doing?",
      sender: 'user',
      timestamp: new Date(Date.now() - 120000),
      type: 'general'
    },
    {
      id: "3",
      content: "Your model scored 78%! It's pretty good but needs some work on risk monitoring. Want me to explain more?",
      sender: 'assistant',
      timestamp: new Date(Date.now() - 60000),
      type: 'analysis'
    }
  ]);

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      type: inputMessage.startsWith('[') ? 'command' : 'general'
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage("");

    // Simulate assistant response
    setTimeout(() => {
      const assistantResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: getSimpleResponse(inputMessage),
        sender: 'assistant',
        timestamp: new Date(),
        type: 'analysis'
      };
      setMessages(prev => [...prev, assistantResponse]);
    }, 1000);
  };

  const getSimpleResponse = (userMessage: string): string => {
    if (userMessage.toLowerCase().includes('compliance') || userMessage.toLowerCase().includes('check')) {
      return "Your model scored 78%! It follows most rules but needs work on risk monitoring. Want me to show you what to fix?";
    } else if (userMessage.toLowerCase().includes('consistency')) {
      return "Your model is pretty stable! It works the same way 87% of the time. That's good!";
    } else if (userMessage.toLowerCase().includes('challenge') || userMessage.toLowerCase().includes('test')) {
      return "I can make tough tests for your model! We have 12 people ready to check your work. Should I start?";
    } else {
      return "I can help you check your model, test how well it works, or answer questions. What would you like to do?";
    }
  };

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (isMinimized) {
    return (
      <Card className="mb-6 shadow-xl bg-white border-2 border-[#085280]/20">
        <CardHeader className="py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-[#085280] rounded-full flex items-center justify-center">
                <MessageCircle className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-xl text-[#085280]">💬 Chat Helper</CardTitle>
                <Badge variant="secondary" className="bg-[#238196]/10 text-[#238196] border border-[#238196]/20">
                  {messages.length} messages
                </Badge>
              </div>
            </div>
            <Button
              variant="ghost"
              size="lg"
              onClick={() => setIsMinimized(false)}
              className="text-[#085280] hover:bg-[#085280]/10"
            >
              <Maximize2 className="h-6 w-6" />
            </Button>
          </div>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="mb-6 shadow-xl bg-white border-2 border-[#085280]/20">
      <CardHeader className="py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-[#085280] rounded-full flex items-center justify-center">
              <MessageCircle className="h-6 w-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl text-[#085280]">💬 Chat Helper</CardTitle>
              <Badge variant="secondary" className="bg-green-100 text-green-700 border border-green-200">
                Online & Ready!
              </Badge>
            </div>
          </div>
          <Button
            variant="ghost"
            size="lg"
            onClick={() => setIsMinimized(true)}
            className="text-[#085280] hover:bg-[#085280]/10"
          >
            <Minimize2 className="h-6 w-6" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Chat Messages */}
          <ScrollArea className="h-48 w-full">
            <div className="space-y-4 pr-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.sender === 'assistant' && (
                    <div className="w-10 h-10 bg-gradient-to-r from-[#085280] to-[#238196] rounded-full flex items-center justify-center flex-shrink-0">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                  )}
                  <div className={`max-w-[80%] ${message.sender === 'user' ? 'order-1' : ''}`}>
                    <div
                      className={`p-4 rounded-xl text-base ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-r from-[#085280] to-[#238196] text-white'
                          : 'bg-gray-50 border-2 border-gray-200'
                      }`}
                    >
                      <p>{message.content}</p>
                    </div>
                    <div className={`text-sm text-muted-foreground mt-1 ${
                      message.sender === 'user' ? 'text-right' : 'text-left'
                    }`}>
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                  {message.sender === 'user' && (
                    <div className="w-10 h-10 bg-[#C55422] rounded-full flex items-center justify-center flex-shrink-0">
                      <User className="h-5 w-5 text-white" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="flex gap-3">
            <Input
              placeholder="Ask me anything about your model..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-1 h-12 text-lg border-2 border-[#085280]/30 focus:border-[#085280] focus:ring-[#085280]/20"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim()}
              size="lg"
              className="h-12 px-6 bg-gradient-to-r from-[#085280] to-[#238196] hover:from-[#085280]/90 hover:to-[#238196]/90"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>

          {/* Quick Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Button
              variant="outline"
              onClick={() => setInputMessage("How is my model doing?")}
              className="h-12 text-base border-2 border-[#085280]/30 text-[#085280] hover:bg-[#085280]/10"
            >
              📊 Check Score
            </Button>
            <Button
              variant="outline"
              onClick={() => setInputMessage("Is my model consistent?")}
              className="h-12 text-base border-2 border-[#238196]/30 text-[#238196] hover:bg-[#238196]/10"
            >
              🔄 Check Consistency
            </Button>
            <Button
              variant="outline"
              onClick={() => setInputMessage("Make new tests for my model")}
              className="h-12 text-base border-2 border-[#C55422]/30 text-[#C55422] hover:bg-[#C55422]/10"
            >
              🎯 Make Tests
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}