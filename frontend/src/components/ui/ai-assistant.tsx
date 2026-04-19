import React, { useState } from "react";
import { Send, Sparkles, X, Loader2, Bell, Check } from "lucide-react";

interface Message {
  text: string;
  isUser: boolean;
  monitorSuggestion?: MonitorSuggestion | null;
}

interface MonitorSuggestion {
  ticker: string;
  target_price: number;
  direction: "above" | "below";
}

interface MonitorDraft {
  ticker: string;
  target_price: string;
  direction: "above" | "below";
  user_email: string;
}

interface AIAssistantProps {
  apiKey?: string;
  onSendMessage?: (message: string) => Promise<{ response: string; monitor_suggestion?: MonitorSuggestion | null }>;
  onCreateMonitor?: (monitor: { ticker: string; target_price: number; direction: string; user_email?: string }) => Promise<void>;
  openAlert?: boolean;
  onCloseAlert?: () => void;
  userEmail?: string;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ apiKey, onSendMessage, onCreateMonitor, openAlert, onCloseAlert, userEmail }) => {
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [isFocused, setIsFocused] = useState<boolean>(false);
  const [monitorDraft, setMonitorDraft] = useState<MonitorDraft | null>(null);
  const [monitorSubmitting, setMonitorSubmitting] = useState<boolean>(false);
  const [monitorSuccess, setMonitorSuccess] = useState<boolean>(false);

  // Open blank alert popup when parent sets openAlert=true
  React.useEffect(() => {
    if (openAlert) {
      setMonitorDraft({ ticker: "", target_price: "", direction: "above", user_email: userEmail || "" });
      setMonitorSuccess(false);
    }
  }, [openAlert, userEmail]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (input.trim() === "" || isTyping) return;
    
    const userMessage = input;
    setMessages((prev) => [...prev, { text: userMessage, isUser: true }]);
    setInput("");
    setIsTyping(true);
    
    try {
      if (onSendMessage) {
        const result = await onSendMessage(userMessage);
        setMessages((prev) => [...prev, {
          text: result.response,
          isUser: false,
          monitorSuggestion: result.monitor_suggestion || null,
        }]);
      } else {
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

  const closeMonitorModal = () => {
    setMonitorDraft(null);
    onCloseAlert?.();
  };

  const handleMonitorSubmit = async () => {
    if (!monitorDraft || !onCreateMonitor) return;
    const price = parseFloat(monitorDraft.target_price);
    if (isNaN(price) || price <= 0) return;

    setMonitorSubmitting(true);
    try {
      await onCreateMonitor({
        ticker: monitorDraft.ticker.toUpperCase(),
        target_price: price,
        direction: monitorDraft.direction,
        user_email: monitorDraft.user_email.trim() || undefined,
      });
      setMonitorSuccess(true);
      setTimeout(() => {
        setMonitorDraft(null);
        setMonitorSuccess(false);
        onCloseAlert?.();
      }, 1500);
    } catch {
      // keep modal open so user can retry
    } finally {
      setMonitorSubmitting(false);
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
                  {!msg.isUser && msg.monitorSuggestion && (
                    <button
                      onClick={() => {
                        setMonitorDraft({
                          ticker: msg.monitorSuggestion!.ticker,
                          target_price: String(msg.monitorSuggestion!.target_price),
                          direction: msg.monitorSuggestion!.direction,
                          user_email: userEmail || "",
                        });
                        setMonitorSuccess(false);
                      }}
                      className="mt-3 inline-flex items-center space-x-1 text-indigo-600 hover:text-indigo-800 text-sm font-medium underline underline-offset-2 cursor-pointer"
                    >
                      <Bell className="h-3.5 w-3.5" />
                      <span>Create a price alert for {msg.monitorSuggestion.ticker}</span>
                    </button>
                  )}
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

      {/* Monitor Suggestion Popup */}
      {monitorDraft && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl border border-indigo-200 w-full max-w-md mx-4 animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-indigo-100">
              <div className="flex items-center space-x-2">
                <Bell className="h-5 w-5 text-indigo-600" />
                <h3 className="text-lg font-semibold text-slate-800">Create Price Alert</h3>
              </div>
              <button onClick={closeMonitorModal} className="text-slate-400 hover:text-slate-600">
                <X className="h-5 w-5" />
              </button>
            </div>

            {monitorSuccess ? (
              <div className="flex flex-col items-center justify-center py-10 space-y-2">
                <Check className="h-12 w-12 text-green-500" />
                <p className="text-green-700 font-medium">Alert created!</p>
              </div>
            ) : (
              <>
                <div className="px-6 py-4 space-y-4">
                  {/* Ticker */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Ticker</label>
                    <input
                      type="text"
                      value={monitorDraft.ticker}
                      onChange={(e) => setMonitorDraft({ ...monitorDraft, ticker: e.target.value.toUpperCase() })}
                      className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  {/* Target Price */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Target Price ($)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={monitorDraft.target_price}
                      onChange={(e) => setMonitorDraft({ ...monitorDraft, target_price: e.target.value })}
                      className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  {/* Direction */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Alert when price goes</label>
                    <select
                      value={monitorDraft.direction}
                      onChange={(e) => setMonitorDraft({ ...monitorDraft, direction: e.target.value as "above" | "below" })}
                      className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-slate-800 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="above">Above target</option>
                      <option value="below">Below target</option>
                    </select>
                  </div>
                  {/* Email (optional) */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Email for notification (optional)</label>
                    <input
                      type="email"
                      value={monitorDraft.user_email}
                      onChange={(e) => setMonitorDraft({ ...monitorDraft, user_email: e.target.value })}
                      placeholder="you@example.com"
                      className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center justify-end space-x-3 px-6 pb-5 pt-2">
                  <button
                    onClick={closeMonitorModal}
                    className="px-4 py-2 text-sm rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleMonitorSubmit}
                    disabled={monitorSubmitting || !monitorDraft.ticker || !monitorDraft.target_price}
                    className="px-4 py-2 text-sm rounded-lg text-white bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    {monitorSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Bell className="h-4 w-4" />}
                    <span>{monitorSubmitting ? "Creating..." : "Create Alert"}</span>
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AIAssistant;
