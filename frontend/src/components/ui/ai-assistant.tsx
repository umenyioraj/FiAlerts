import React, { useState } from "react";
import { Send, Sparkles, X, Loader2 } from "lucide-react";

interface Message {
  text: string;
  isUser: boolean;
}

interface AIAssistantProps {
  apiKey?: string;
  onSendMessage?: (message: string) => Promise<string>;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ apiKey, onSendMessage }) => {
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [isFocused, setIsFocused] = useState<boolean>(false);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (input.trim() === "" || isTyping) return;
    
    const userMessage = input;
    setMessages((prev) => [...prev, { text: userMessage, isUser: true }]);
    setInput("");
    setIsTyping(true);
    
    try {
      if (onSendMessage) {
        const response = await onSendMessage(userMessage);
        setMessages((prev) => [...prev, { text: response, isUser: false }]);
      } else {
        // Fallback simulation if no handler provided
        setTimeout(() => {
          const response = "Hi there! I'm your AI assistant. How can I help you today?";
          setMessages((prev) => [...prev, { text: response, isUser: false }]);
          setIsTyping(false);
        }, 1500);
        return;
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { text: "Sorry, I encountered an error. Please try again.", isUser: false }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="w-full max-w-4xl mx-auto h-[600px] bg-white/90 backdrop-blur-sm rounded-xl overflow-hidden shadow-2xl border border-indigo-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-4 border-b border-indigo-300 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Sparkles className="text-white h-5 w-5" />
          <h2 className="text-white font-semibold text-lg">FiAlerts AI Assistant</h2>
        </div>
        {messages.length > 0 && (
          <button 
            onClick={clearChat}
            className="text-white/80 hover:text-white transition-colors"
            title="Clear conversation"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
      
      {/* Messages container */}
      <div className="p-6 h-[calc(100%-140px)] overflow-y-auto bg-gradient-to-b from-white/50 to-indigo-50/30">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Sparkles className="h-16 w-16 text-indigo-500 mb-4" />
            <h3 className="text-indigo-900 text-2xl font-semibold mb-2">How can I help you today?</h3>
            <p className="text-slate-600 text-sm max-w-md">
              Ask me about stocks, financial analysis, or market trends. I'm here to provide insights!
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] p-4 rounded-2xl shadow-md ${
                    msg.isUser
                      ? "bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-tr-sm"
                      : "bg-white text-slate-800 rounded-tl-sm border border-indigo-100"
                  } animate-fade-in`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="max-w-[80%] p-4 rounded-2xl bg-white text-slate-800 rounded-tl-sm border border-indigo-100 shadow-md">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></div>
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse delay-75"></div>
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse delay-150"></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Input form */}
      <form 
        onSubmit={handleSubmit}
        className={`p-4 border-t ${isFocused ? 'border-indigo-400 bg-indigo-50/50' : 'border-indigo-200 bg-white'} transition-colors duration-200`}
      >
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Ask about stocks, financial analysis, or market trends..."
            disabled={isTyping}
            className="w-full bg-white border border-indigo-300 rounded-full py-3 pl-5 pr-12 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={input.trim() === "" || isTyping}
            className={`absolute right-1.5 rounded-full p-2.5 ${
              input.trim() === "" || isTyping
                ? "text-slate-400 bg-slate-100 cursor-not-allowed"
                : "text-white bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-md"
            } transition-all duration-200`}
          >
            {isTyping ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AIAssistant;
