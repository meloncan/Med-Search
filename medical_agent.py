import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform
import contextlib
from typing import TypedDict, Annotated, List
from functools import partial

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

# 필요한 라이브러리 import
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# MCP-Adapter 관련 라이브러리 import
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage

from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)


logging.langsmith("Medical Search")

# 페이지 설정
st.set_page_config(
    page_title="🏥 의료 논문 검색 에이전트", 
    page_icon="🏥", 
    layout="wide"
)

# config.json 파일 경로 설정
CONFIG_FILE_PATH = "medical_config.json"

# --- 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 유틸리티 함수들 ---
def random_uuid():
    """랜덤 UUID 생성"""
    import uuid
    return str(uuid.uuid4())

def convert_history_to_messages(history, max_turns=8):
    """
    st.session_state.history를 LangChain 메시지로 변환합니다.
    
    Args:
        history: 대화 기록 리스트
        max_turns: 최대 포함할 대화 턴 수
    
    Returns:
        List[BaseMessage]: LangChain 메시지 리스트
    """
    messages = []
    
    # Medical workflow용 시스템 프롬프트 추가
    system_prompt = """당신은 의료 관련 질문에 답변하는 전문적이고 도움이 되는 AI 어시스턴트입니다.

주요 역할:
- 의료/과학 논문 검색 및 연구 정보 제공
- 의료 정보에 대한 정확하고 신뢰할 수 있는 답변 제공
- 이전 대화 내용을 기억하고 맥락을 고려한 연속적인 대화
- 복잡한 의학 용어를 이해하기 쉽게 설명
- 필요시 전문의 상담을 권유

사용 가능한 도구들:
- PubMed 검색: 의학/과학 논문 검색 및 분석
- 기타 의료 관련 데이터베이스

중요 사항:
- 진단이나 치료를 직접 제공하지 않고, 일반적인 정보만 제공
- 응급 상황 시 즉시 의료진에게 연락하도록 안내
- 이전 대화 내용을 참고하여 연관성 있는 답변 제공
- 논문 검색 시 영어 번역을 통해 정확한 검색 수행"""
    
    messages.append(SystemMessage(content=system_prompt))
    
    # 최근 N턴만 포함 (토큰 제한 고려)
    # assistant_tool 메시지는 제외하고 user와 assistant만 포함
    filtered_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
    recent_history = filtered_history[-(max_turns*2):] if len(filtered_history) > max_turns*2 else filtered_history
    
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    return messages

def get_conversation_context_summary(history, max_context_turns=3):
    """
    번역을 위한 대화 맥락 요약을 생성합니다.
    
    Args:
        history: 대화 기록
        max_context_turns: 포함할 최대 맥락 턴 수
    
    Returns:
        str: 맥락 요약 텍스트
    """
    if not history:
        return ""
    
    # 최근 대화만 추출 (assistant_tool 제외)
    filtered_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
    recent_turns = filtered_history[-(max_context_turns*2):] if len(filtered_history) > max_context_turns*2 else filtered_history
    
    if not recent_turns:
        return ""
    
    context_parts = []
    for msg in recent_turns:
        if msg["role"] == "user":
            context_parts.append(f"사용자: {msg['content'][:100]}...")  # 첫 100자만
        elif msg["role"] == "assistant":
            context_parts.append(f"AI: {msg['content'][:100]}...")  # 첫 100자만
    
    return "\n".join(context_parts)

def load_config_from_json():
    """config.json 파일에서 설정을 로드합니다."""
    default_config = {
        "pubmed": {
            "command": "cmd",
            "args": [
                "/c",
                "npx",
                "-y",
                "@smithery/cli@latest",
                "run",
                "@JackKuo666/pubmed-mcp-server",
                "--key",
                "a0fde5b8-88e9-46d3-ab62-ad5096bd7d4b",
            ],
            "transport": "stdio"
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
        return default_config

def save_config_to_json(config):
    """설정을 config.json 파일에 저장합니다."""
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"설정 파일 저장 중 오류 발생: {str(e)}")
        return False

def get_tool_description_from_available_tools(tool_name: str) -> str | None:
    """연결된 도구들에서 실제 description을 가져옵니다."""
    if not st.session_state.get('available_tools'):
        return None
    
    # 연결된 도구들에서 해당 이름의 도구 찾기
    for tool in st.session_state.available_tools:
        if tool.name.lower() == tool_name.lower() or tool_name.lower() in tool.name.lower():
            if hasattr(tool, 'description') and tool.description and tool.description.strip():
                return tool.description.strip()
    
    return None  # 설명이 없는 경우

# --- LangGraph 노드 함수들 ---
async def translate_to_english(state: AgentState, translator_model):
    """한글 사용자 입력을 영어로 번역합니다 (이전 대화 맥락 고려)."""
    messages = state["messages"]
    last_human_message = None
    
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        return {"messages": []}
    
    # 이전 대화 맥락 가져오기
    context_summary = get_conversation_context_summary(st.session_state.history)
    
    if context_summary:
        translation_prompt = f"""
        이전 대화 맥락:
        {context_summary}

        현재 질문을 위 맥락을 고려하여 의료/과학 논문 검색에 적합한 영어로 번역해주세요.
        의학 용어는 정확한 영어 용어로 번역하고, 검색 키워드로 적합하도록 간결하게 번역해주세요.
        이전 대화와의 연관성을 고려하여 번역하되, 검색에 필요한 핵심 키워드를 포함해주세요.

        현재 한글 질문: {last_human_message.content}

        영어 번역:"""
    else:
        translation_prompt = f"""
        다음 한글 텍스트를 의료/과학 논문 검색에 적합한 영어로 번역해주세요. 
        의학 용어는 정확한 영어 용어로 번역하고, 검색 키워드로 적합하도록 간결하게 번역해주세요.

        한글 텍스트: {last_human_message.content}

        영어 번역:"""

    response = await translator_model.ainvoke([HumanMessage(content=translation_prompt)])
    translated_text = response.content.strip()
    
    translated_message = HumanMessage(content=translated_text)
    
    new_messages = []
    for message in messages[:-1]:
        new_messages.append(message)
    new_messages.append(translated_message)
    
    return {"messages": new_messages}

async def call_model(state: AgentState, model):
    """LLM을 호출하여 응답 또는 도구 호출을 생성합니다."""
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def translate_to_korean(state: AgentState, translator_model):
    """영어 AI 응답을 한글로 번역합니다."""
    messages = state["messages"]
    
    # 디버깅: 메시지 상태 확인
    print(f"🔍 번역 단계 - 총 메시지 수: {len(messages)}")
    for i, msg in enumerate(messages[-5:]):  # 최근 5개 메시지만 출력
        print(f"  {i}: {type(msg).__name__} - {str(msg)[:100]}...")
    
    last_tool_message = None
    last_ai_message = None
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and last_tool_message is None:
            last_tool_message = message
            print(f"✅ ToolMessage 찾음: {str(message.content)[:100]}...")
        elif isinstance(message, AIMessage) and not message.tool_calls and last_ai_message is None:
            last_ai_message = message
            print(f"✅ AIMessage 찾음: {str(message.content)[:100]}...")
    
    content_to_translate = None
    translation_source = None
    
    if last_tool_message and last_tool_message.content:
        content_to_translate = last_tool_message.content
        translation_source = "ToolMessage"
    elif last_ai_message and last_ai_message.content:
        content_to_translate = last_ai_message.content
        translation_source = "AIMessage"
    
    print(f"🎯 번역할 내용: {translation_source} - {str(content_to_translate)[:200] if content_to_translate else 'None'}...")
    
    if content_to_translate is None:
        print("❌ 번역할 내용이 없음 - 빈 메시지 반환")
        # 빈 메시지 대신 오류 메시지 반환
        error_message = AIMessage(content="죄송합니다. 번역할 내용을 찾을 수 없습니다. 다시 시도해 주세요.")
        return {"messages": [error_message]}
    
    print("🔄 번역 시작...")
    
    translation_prompt = f"""
    다음 영어 텍스트를 한국어로 번역해주세요. 의료/과학 논문 정보의 경우 다음 규칙을 따라주세요:

    번역 규칙:
    1. PMID, DOI 등의 식별자: 그대로 유지
    2. Title (제목): 영어 이름 그대로 유지
    3. Authors (저자): 영어 이름 그대로 유지 
    4. Journal (저널명): 영어 그대로 유지
    5. Publication Date: 영어 그대로 유지
    6. Abstract (초록): 한국어로 상세히 번역
    7. 기타 메타데이터: 적절히 번역하되 정보 손실 없이 보존

    번역할 텍스트:
    {content_to_translate}

    한국어 번역:"""

    try:
        response = await translator_model.ainvoke([HumanMessage(content=translation_prompt)])
        translated_text = response.content.strip()
        
        print(f"✅ 번역 완료: {translated_text[:200]}...")
        
        translated_message = AIMessage(content=translated_text)
        return {"messages": [translated_message]}
        
    except Exception as e:
        print(f"❌ 번역 중 오류: {str(e)}")
        error_message = AIMessage(content=f"번역 중 오류가 발생했습니다: {str(e)}")
        return {"messages": [error_message]}

async def korean_direct_answer(state: AgentState, translator_model):
    """도구 사용 없이 바로 한글로 답변합니다 - 멀티턴 대화 지원."""
    messages = state["messages"]
    
    # 이전 대화 맥락을 고려한 프롬프트 생성
    context_summary = get_conversation_context_summary(st.session_state.history, max_context_turns=5)
    
    last_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        return {"messages": []}
    
    if context_summary:
        korean_prompt = f"""
        이전 대화 맥락:
        {context_summary}

        위 맥락을 고려하여 다음 질문에 대해 한국어로 답변해주세요. 
        이전 대화와의 연관성을 고려하고, 도구 사용이 필요하지 않은 일반적인 질문이므로 당신의 지식을 바탕으로 정확하고 자세한 답변을 제공해주세요.

        현재 질문: {last_human_message.content}

        한국어 답변:"""
    else:
        korean_prompt = f"""
        다음 질문에 대해 한국어로 직접 답변해주세요. 도구 사용이 필요하지 않은 일반적인 질문이므로 당신의 지식을 바탕으로 정확하고 자세한 답변을 제공해주세요.

        질문: {last_human_message.content}

        한국어 답변:"""

    response = await translator_model.ainvoke([HumanMessage(content=korean_prompt)])
    korean_answer = response.content.strip()
    
    korean_message = AIMessage(content=korean_answer)
    
    return {"messages": [korean_message]}

# --- 워크플로우 분류 및 관리 ---
class LLMWorkflowClassifier:
    """LLM 기반 지능적 워크플로우 분류기 - 도구 description을 읽고 reasoning으로 판단"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.cache = {}  # 성능 최적화용 캐시
    
    def extract_tool_descriptions(self, tools: list) -> list:
        """도구들의 상세 정보를 추출합니다."""
        tool_info = []
        
        for tool in tools:
            try:
                # 도구의 기본 정보 추출
                info = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', '설명 없음'),
                }
                
                # args_schema에서 추가 정보 추출 (선택적)
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    schema = tool.args_schema
                    if hasattr(schema, '__annotations__'):
                        info["parameters"] = list(schema.__annotations__.keys())
                    elif hasattr(schema, 'model_fields'):
                        info["parameters"] = list(schema.model_fields.keys())
                
                tool_info.append(info)
                
            except Exception as e:
                # 정보 추출 실패 시 기본 정보만 포함
                tool_info.append({
                    "name": tool.name,
                    "description": "정보 추출 실패"
                })
        
        return tool_info
    
    def format_tools_for_llm(self, tool_info: list) -> str:
        """도구 정보를 LLM이 읽기 쉬운 형태로 포맷팅합니다."""
        formatted = []
        
        for i, tool in enumerate(tool_info, 1):
            tool_desc = f"{i}. **{tool['name']}**"
            tool_desc += f"\n   - 설명: {tool['description']}"
            
            if "parameters" in tool and tool["parameters"]:
                tool_desc += f"\n   - 매개변수: {', '.join(tool['parameters'])}"
            
            formatted.append(tool_desc)
        
        return "\n\n".join(formatted)
    
    async def classify_workflow(self, query: str, tools: list) -> dict:
        """LLM을 사용하여 워크플로우 타입을 지능적으로 분류합니다."""
        
        # 1. 캐시 키 생성
        cache_key = f"{query}_{len(tools)}_{hash(str([t.name for t in tools]))}"
        
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            print(f"🚀 캐시된 결과 사용: {cached_result}")
            return cached_result
        
        # 2. 도구 정보 추출
        tool_info = self.extract_tool_descriptions(tools)
        formatted_tools = self.format_tools_for_llm(tool_info)
        
        # 3. LLM 분류 프롬프트 생성
        classification_prompt = f"""당신은 사용자 질문을 분석하여 최적의 워크플로우를 선택하는 전문가입니다.

        **사용자 질문**: "{query}"

        **사용 가능한 도구들**:
        {formatted_tools}

        **워크플로우 옵션**:

1. **medical** (매우 제한적으로 사용):
   - **오직 의료/과학 논문 검색과 연구 분석에만 사용**
   - PubMed, DOI, PMID 관련 논문 검색
   - 특정 연구 논문의 내용 분석이나 메타분석
   - 학술적 연구 데이터 조회
   - 영어 번역 과정이 포함되어 느린 응답
   - 예: "당뇨병 관련 최신 논문 검색", "PMID 12345 논문 내용", "고혈압 치료 관련 연구 동향"

2. **general** (기본 선택):
   - **의료 관련 질문이라도 논문 검색이 아니면 여기 사용**
   - 일반적인 의료 정보, 증상, 치료법, 건강 상식
   - 실시간 의료 뉴스, 병원 정보, 약물 정보
   - 웹 검색을 통한 의료 정보 조회
   - 빠른 응답과 다양한 도구 활용 가능
   - 뉴스, 날씨, 시간, 계산, 일반 검색 등
   - 예: "감기 증상과 치료법", "근처 병원 찾기", "혈압약 부작용", "건강한 식단", "의료 뉴스"

        **중요한 분류 기준**:
        1. **논문/연구 검색 여부**: 학술 논문이나 연구 데이터를 찾는 질문인가?
        2. **PMID/DOI 언급**: 특정 논문 식별자가 포함되어 있는가?
        3. **연구 동향/메타분석**: 특정 주제의 연구 동향이나 분석을 요구하는가?
        
        **주의사항**:
        - 의료 관련 질문이라도 논문 검색이 아니면 **반드시 general 선택**
        - 일반적인 의료 정보나 건강 상식은 general 워크플로우가 더 적합
        - 확실하지 않으면 general을 선택 (더 빠르고 유연함)

        다음 JSON 형식으로만 답변해주세요:
        {{
            "workflow": "medical" 또는 "general",
            "confidence": 0.0-1.0 사이의 확신도,
            "reason": "선택한 이유에 대한 간단한 설명"
        }}"""

        try:
            # 4. LLM 호출
            print(f"🧠 LLM 워크플로우 분류 시작...")
            
            response = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content=classification_prompt)]),
                timeout=15.0  # 15초 타임아웃
            )
            
            # 5. 응답 파싱
            result = self.parse_llm_response(response.content)
            
            # 6. 캐싱
            self.cache[cache_key] = result
            
            print(f"🎯 LLM 분류 결과: {result}")
            return result
            
        except asyncio.TimeoutError:
            print("⏰ LLM 분류 타임아웃 - 기본값 사용")
            return self.get_fallback_classification(query, tool_info)
        except Exception as e:
            print(f"❌ LLM 분류 오류: {str(e)} - 기본값 사용")
            return self.get_fallback_classification(query, tool_info)
    
    def parse_llm_response(self, response_content: str) -> dict:
        """LLM 응답을 파싱하여 구조화된 결과로 변환합니다."""
        try:
            import json
            import re
            
            # JSON 부분만 추출
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 필수 필드 검증
                if "workflow" in result and result["workflow"] in ["medical", "general"]:
                    return {
                        "workflow": result["workflow"],
                        "confidence": result.get("confidence", 0.8),
                        "reason": result.get("reason", "LLM 분류"),
                        "method": "llm"
                    }
            
            # 파싱 실패 시 텍스트에서 키워드 찾기
            if "medical" in response_content.lower():
                return {"workflow": "medical", "confidence": 0.7, "reason": "텍스트 파싱", "method": "fallback"}
            else:
                return {"workflow": "general", "confidence": 0.7, "reason": "텍스트 파싱", "method": "fallback"}
                
        except Exception as e:
            print(f"⚠️ LLM 응답 파싱 실패: {str(e)}")
            return {"workflow": "general", "confidence": 0.5, "reason": "파싱 실패", "method": "error"}
    
    def get_fallback_classification(self, query: str, tool_info: list) -> dict:
        """LLM 분류 실패 시 사용할 간단한 fallback 로직"""
        # LLM 분류 실패 시 안전하게 general 워크플로우를 기본값으로 사용
        return {"workflow": "general", "confidence": 0.5, "reason": "LLM 분류 실패로 기본값 사용", "method": "fallback"}

# 하위 호환성을 위한 기존 인터페이스 유지
class WorkflowClassifier:
    """기존 WorkflowClassifier의 호환성 래퍼 - 이제 LLM reasoning 사용"""
    
    _llm_classifier = None
    
    @staticmethod
    async def initialize_llm_classifier():
        """LLM 분류기 초기화"""
        if WorkflowClassifier._llm_classifier is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            WorkflowClassifier._llm_classifier = LLMWorkflowClassifier(llm)
    
    @staticmethod
    async def select_workflow_type(query: str, tools: list) -> str:
        """🧠 LLM 기반 워크플로우 선택 - 도구 description을 읽고 reasoning으로 판단"""
        await WorkflowClassifier.initialize_llm_classifier()
        
        try:
            result = await WorkflowClassifier._llm_classifier.classify_workflow(query, tools)
            print(f"🎯 최종선택: {result['workflow']} (확신도: {result['confidence']:.2f}, 방법: {result['method']}, 이유: {result['reason']})")
            return result["workflow"]
        except Exception as e:
            print(f"❌ LLM 분류 실패, 극단적 fallback 사용: {str(e)}")
            # 극단적 fallback - 안전하게 general 기본값 사용
            return "general"

class WorkflowFactory:
    """워크플로우를 생성하는 팩토리 클래스"""
    
    @staticmethod
    def create_medical_workflow(tools: list, model, translator_model, mcp_config):
        """의료 전용 워크플로우 생성 (기존 로직)"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("translate_to_english", partial(translate_to_english, translator_model=translator_model))
        workflow.add_node("agent", partial(call_model, model=model.bind_tools(tools)))
        
        # 안전한 도구 노드
        async def safe_tool_node(state):
            try:
                async with connect_all_mcp_servers(mcp_config) as (fresh_tools, _):
                    if fresh_tools:
                        tool_node = ToolNode(fresh_tools)
                        return await tool_node.ainvoke(state)
            except Exception as e:
                error_msg = f"도구 실행 중 오류 발생: {str(e)}"
                return {"messages": [ToolMessage(content=error_msg, tool_call_id="error")]}
        
        workflow.add_node("action", safe_tool_node)
        workflow.add_node("translate_to_korean", partial(translate_to_korean, translator_model=translator_model))
        workflow.add_node("direct_answer", partial(korean_direct_answer, translator_model=translator_model))

        # 워크플로우 연결
        workflow.set_entry_point("translate_to_english")
        workflow.add_edge("translate_to_english", "agent")
        workflow.add_conditional_edges(
            "agent", 
            should_continue_or_answer_medical, 
            {"continue": "action", "direct_answer": "direct_answer"}
        )
        workflow.add_conditional_edges(
            "action", after_action_medical,
            {"agent": "agent", "translate": "translate_to_korean"}
        )
        workflow.add_edge("translate_to_korean", END)
        workflow.add_edge("direct_answer", END)
        
        return workflow
    
    @staticmethod  
    def create_general_workflow(tools: list, model, translator_model, mcp_config):
        """일반 도구용 워크플로우 생성 (번역 없음)"""
        workflow = StateGraph(AgentState)
        
        # 한글 직접 처리 에이전트 (멀티턴 지원)
        async def korean_agent(state):
            """한글로 직접 처리하는 에이전트 - 멀티턴 대화 지원"""
            messages = state["messages"]
            
            # General workflow용 시스템 프롬프트가 없다면 추가
            has_system_prompt = any(isinstance(msg, SystemMessage) for msg in messages)
            
            if not has_system_prompt:
                general_system_prompt = """당신은 다양한 정보를 제공하는 전문적이고 도움이 되는 AI 어시스턴트입니다.

                주요 역할:
                - 실시간 정보 검색 및 제공 (뉴스, 날씨, 시간 등)
                - 일반적인 질문에 대한 정확하고 유용한 답변
                - 이전 대화 내용을 기억하고 맥락을 고려한 연속적인 대화
                - 사용자의 요청에 따라 적절한 도구 활용

                사용 가능한 도구들:
                - 웹 검색: 최신 뉴스, 블로그, 커뮤니티 등 다양한 정보 검색
                - 날씨 정보: 현재 날씨 및 예보 정보
                - 시간 조회: 현재 시간 및 시간대 정보
                - 기타 실시간 데이터 조회

                중요 사항:
                - 이전 대화 내용을 참고하여 연관성 있는 답변 제공
                - 실시간 정보가 필요한 경우 적절한 도구 사용
                - 정확하고 최신의 정보 제공에 중점
                - 연결된 도구들의 기능을 최대한 활용"""
                
                # 시스템 프롬프트를 맨 앞에 추가
                messages = [SystemMessage(content=general_system_prompt)] + messages
                state = {"messages": messages}
            
            return await call_model(state, model.bind_tools(tools))
        
        # 간단한 도구 노드
        async def simple_tool_node(state):
            try:
                async with connect_all_mcp_servers(mcp_config) as (fresh_tools, _):
                    if fresh_tools:
                        tool_node = ToolNode(fresh_tools)
                        return await tool_node.ainvoke(state)
            except Exception as e:
                error_msg = f"도구 실행 중 오류 발생: {str(e)}"
                return {"messages": [ToolMessage(content=error_msg, tool_call_id="error")]}
        
        # 한글 응답 정리
        async def format_korean_response(state):
            """한글 응답을 정리하는 노드"""
            messages = state["messages"]
            
            # 마지막 ToolMessage나 AIMessage 찾기
            last_content = None
            for message in reversed(messages):
                if isinstance(message, ToolMessage) and message.content:
                    last_content = message.content
                    break
                elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    last_content = message.content
                    break
            
            if last_content:
                # 간단한 포맷팅 (필요시)
                formatted_content = last_content
                return {"messages": [AIMessage(content=formatted_content)]}
            else:
                return {"messages": [AIMessage(content="죄송합니다. 응답을 생성할 수 없습니다.")]}
        
        workflow.add_node("agent", korean_agent)
        workflow.add_node("action", simple_tool_node)
        workflow.add_node("format_response", format_korean_response)
        workflow.add_node("direct_answer", partial(korean_direct_answer, translator_model=translator_model))

        # 워크플로우 연결 (단순화)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue_or_answer_general,
            {"continue": "action", "direct_answer": "direct_answer"}
        )
        workflow.add_edge("action", "format_response")
        workflow.add_edge("format_response", END)
        workflow.add_edge("direct_answer", END)
        
        return workflow

# --- 조건부 엣지 로직 ---
def should_continue_or_answer_medical(state: AgentState) -> str:
    """의료 워크플로우용 - LLM의 응답에 따라 다음 노드를 결정합니다."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "direct_answer"

def should_continue_or_answer_general(state: AgentState) -> str:
    """일반 워크플로우용 - LLM의 응답에 따라 다음 노드를 결정합니다."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "direct_answer"

def after_action_medical(state: AgentState) -> str:
    """의료 워크플로우용 - Action 후에는 항상 번역으로 이동합니다."""
    return "translate"

# General workflow는 after_action이 필요 없음 (바로 format_response로 이동)

# --- MCP 서버 관리 ---
def update_server_status(server_name, status, error_message="", tools_count=0):
    """서버 상태를 업데이트합니다."""
    import time
    st.session_state.server_status[server_name] = {
        "status": status,
        "tools_count": tools_count,
        "error_message": error_message,
        "last_updated": time.time()
    }
    
    # 로그 추가
    log_message = f"[{server_name}] {status}"
    if error_message:
        log_message += f": {error_message}"
    st.session_state.connection_logs.append(log_message)

# 글로벌 연결 관리를 위한 변수
_global_mcp_connections = {}

class MCPConnectionManager:
    """MCP 연결을 안전하게 관리하는 클래스"""
    
    @staticmethod
    async def create_connection(server_name, server_config):
        """새로운 MCP 서버 연결을 생성합니다."""
        update_server_status(server_name, "connecting", "서버 연결 시도 중...")
        
        try:
            params = StdioServerParameters(
                command=server_config['command'],
                args=server_config['args']
            )
            
            # 컨텍스트 매니저를 사용하여 안전하게 연결
            async def _connection_generator():
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        try:
                            await asyncio.wait_for(session.initialize(), timeout=30.0)
                            update_server_status(server_name, "connecting", "도구 로드 중...")
                            
                            tools = await load_mcp_tools(session)
                            tool_count = len(tools)
                            update_server_status(server_name, "connected", f"{tool_count}개 도구 연결 성공", tool_count)
                            
                            # 무한 대기하여 연결 유지
                            yield tools, session
                            
                        except asyncio.TimeoutError:
                            error_msg = "서버 초기화 타임아웃 (30초 초과)"
                            update_server_status(server_name, "failed", error_msg)
                            raise Exception(error_msg)
                        except Exception as e:
                            error_msg = f"세션 초기화 실패: {str(e)}"
                            update_server_status(server_name, "failed", error_msg)
                            raise Exception(error_msg)
            
            return _connection_generator()
            
        except Exception as e:
            error_msg = f"서버 연결 실패: {str(e)}"
            update_server_status(server_name, "failed", error_msg)
            raise Exception(error_msg)

async def connect_all_mcp_servers_simple(mcp_config):
    """간단한 방식으로 MCP 서버에 연결합니다."""
    all_tools = []
    
    # 연결 로그 초기화
    st.session_state.connection_logs = []
    
    try:
        # 기존 방식 사용하되 더 간단하게
        async with connect_all_mcp_servers(mcp_config) as (tools, server_sessions):
            all_tools = tools
            # 도구만 반환하고 연결 관리는 기존 방식 사용
            return all_tools
            
    except Exception as e:
        st.error(f"❌ MCP 서버 연결 실패: {str(e)}")
        return []

# 기존 컨텍스트 매니저 함수는 유지하되 사용하지 않음
@contextlib.asynccontextmanager
async def connect_mcp_server(server_name, server_config):
    """하나의 MCP 서버에 연결하고 도구를 로드합니다."""
    update_server_status(server_name, "connecting", "서버 연결 시도 중...")
    
    try:
        params = StdioServerParameters(
            command=server_config['command'],
            args=server_config['args']
        )
        
        async with stdio_client(params) as (read, write):
            update_server_status(server_name, "connecting", "클라이언트 세션 초기화 중...")
            
            async with ClientSession(read, write) as session:
                try:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                    update_server_status(server_name, "connecting", "도구 로드 중...")
                    
                    tools = await load_mcp_tools(session)
                    tool_count = len(tools)
                    update_server_status(server_name, "connected", f"{tool_count}개 도구 연결 성공", tool_count)
                    
                    yield tools, session
                    
                except asyncio.TimeoutError:
                    error_msg = "서버 초기화 타임아웃 (30초 초과)"
                    update_server_status(server_name, "failed", error_msg)
                    raise Exception(error_msg)
                except Exception as e:
                    error_msg = f"세션 초기화 실패: {str(e)}"
                    update_server_status(server_name, "failed", error_msg)
                    raise Exception(error_msg)
    
    except Exception as e:
        error_msg = f"서버 연결 실패: {str(e)}"
        update_server_status(server_name, "failed", error_msg)
        raise Exception(error_msg)

@contextlib.asynccontextmanager
async def connect_all_mcp_servers(mcp_config):
    """모든 MCP 서버에 동시에 연결하고 모든 도구를 반환합니다."""
    all_tools = []
    server_sessions = {}
    
    # 연결 로그 초기화
    st.session_state.connection_logs = []
    
    async with contextlib.AsyncExitStack() as stack:
        for server_id, config in mcp_config.items():
            try:
                tools, session = await stack.enter_async_context(connect_mcp_server(server_id, config))
                all_tools.extend(tools)
                server_sessions[server_id] = {
                    'session': session,
                    'tools': tools,
                    'name': server_id
                }
            except Exception as e:
                # 상태는 이미 connect_mcp_server에서 업데이트됨
                continue
        
        yield all_tools, server_sessions

# --- 스트리밍 관련 함수들 ---
def get_streaming_callback(text_placeholder, tool_placeholder):
    """스트리밍 콜백 함수를 생성합니다."""
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        # AIMessage 처리 (번역 결과 등)
        if isinstance(message_content, AIMessage):
            if message_content.content and not message_content.tool_calls:
                # 번역된 결과나 직접 답변
                accumulated_text.append(message_content.content)
                text_placeholder.markdown("".join(accumulated_text))
                print(f"📝 AIMessage 처리: {message_content.content[:100]}...")
        
        # AIMessageChunk 처리 (스트리밍)
        elif isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                        st.markdown("".join(accumulated_tool))
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
        
        # ToolMessage 처리
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                st.markdown("".join(accumulated_tool))
        
        return None

    return callback_func, accumulated_text, accumulated_tool

async def astream_graph(graph, inputs, callback=None, config=None):
    """그래프를 스트리밍하여 실행합니다."""
    final_state = None
    async for output in graph.astream(inputs, config=config, stream_mode="values"):
        final_state = output
        if callback:
            callback({"content": final_state["messages"][-1]})
    
    return final_state

# --- 메인 처리 함수들 ---
async def cleanup_mcp_client():
    """기존 MCP 클라이언트를 안전하게 종료합니다."""
    # 간단한 상태 정리만 수행
    if "mcp_connections" in st.session_state:
        st.session_state.mcp_connections = None
    
    # 기타 관련 상태들도 정리
    st.session_state.session_initialized = False
    st.session_state.agent = None

def print_message():
    """채팅 기록을 화면에 출력합니다."""
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(message["content"])
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1

async def process_basic_query(query, text_placeholder, timeout_seconds=120):
    """MCP 도구 없이 기본 모델만으로 질문을 처리합니다 (멀티턴 지원)."""
    try:
        if st.session_state.basic_model:
            with text_placeholder.container():
                # 이전 대화가 있는지 확인
                if st.session_state.history:
                    st.write("🧠 이전 대화를 기억하며 답변 생성 중...")
                else:
                    st.write("🤖 기본 모델로 답변 생성 중...")
            
            # 이전 대화 기록을 포함한 메시지 체인 구성
            max_turns = st.session_state.get('conversation_memory_turns', 8)
            messages = convert_history_to_messages(st.session_state.history, max_turns)
            
            # 현재 사용자 질문 추가
            messages.append(HumanMessage(content=query))
            
            response = await asyncio.wait_for(
                st.session_state.basic_model.ainvoke(messages),
                timeout=timeout_seconds,
            )
            
            final_text = response.content
            return {"success": True}, final_text, ""
        else:
            return (
                {"error": "🚫 기본 모델이 초기화되지 않았습니다."},
                "🚫 기본 모델이 초기화되지 않았습니다.",
                "",
            )
    except asyncio.TimeoutError:
        error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다."
        return {"error": error_msg}, error_msg, ""
    except Exception as e:
        import traceback
        error_msg = f"❌ 기본 모델 처리 중 오류: {str(e)}"
        return {"error": error_msg}, error_msg, ""

async def process_query_smart(query, text_placeholder, tool_placeholder, timeout_seconds=120):
    """스마트 워크플로우 선택으로 사용자 질문을 처리합니다."""
    try:
        if not st.session_state.session_initialized:
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
        
        # 1. 워크플로우 타입 동적 선택 - LLM reasoning 사용
        tools = st.session_state.get('available_tools', [])
        selected_workflow = await WorkflowClassifier.select_workflow_type(query, tools)
        
        # 2. 선택된 워크플로우에 따라 에이전트 생성
        if selected_workflow == "medical":
            agent = WorkflowFactory.create_medical_workflow(
                tools, 
                st.session_state.main_model, 
                st.session_state.translator_model,
                st.session_state.mcp_config
            ).compile(checkpointer=MemorySaver())
            progress_message = "🔄 의료 워크플로우 실행 중 (한글 → 영어 번역 → 검색 → 한글 번역)..."
        else:
            agent = WorkflowFactory.create_general_workflow(
                tools,
                st.session_state.main_model,
                st.session_state.translator_model, 
                st.session_state.mcp_config
            ).compile(checkpointer=MemorySaver())
            progress_message = "🔄 일반 워크플로우 실행 중 (직접 처리)..."
        
        # 3. 스트리밍 콜백 설정
        streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
            get_streaming_callback(text_placeholder, tool_placeholder)
        )
        
        try:
            # 4. 진행 상황 표시
            with text_placeholder.container():
                st.write(progress_message)
            
            # 멀티턴 대화를 위해 이전 대화 기록 포함
            max_turns = st.session_state.get('conversation_memory_turns', 8)
            messages = convert_history_to_messages(st.session_state.history, max_turns)
            
            # 현재 사용자 질문 추가
            messages.append(HumanMessage(content=query))
            
            inputs = {"messages": messages}
            
            # 5. 워크플로우 실행
            response = await asyncio.wait_for(
                astream_graph(
                    agent,
                    inputs,
                    callback=streaming_callback,
                    config=RunnableConfig(
                        recursion_limit=st.session_state.recursion_limit,
                        thread_id=st.session_state.thread_id,
                    ),
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
            return {"error": error_msg}, error_msg, ""

        # 6. 결과 처리
        streaming_text = "".join(accumulated_text_obj)
        final_tool = "".join(accumulated_tool_obj)
        
        # 최종 상태에서 메시지 추출
        final_text = streaming_text
        if response and "messages" in response:
            # 마지막 AIMessage 찾기
            last_ai_message = None
            for message in reversed(response["messages"]):
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    last_ai_message = message
                    break
            
            if last_ai_message:
                final_text = last_ai_message.content
                print(f"✅ 최종 텍스트 추출 ({selected_workflow}): {final_text[:200]}...")
                
                # UI에도 표시
                with text_placeholder.container():
                    st.markdown(final_text)
            elif streaming_text:
                final_text = streaming_text
                print(f"📝 스트리밍 텍스트 사용 ({selected_workflow}): {final_text[:200]}...")
            else:
                print("⚠️ 최종 텍스트를 찾을 수 없음")
                final_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
        
        return response, final_text, final_tool
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

# 기존 함수도 유지 (하위 호환성)
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=120):
    """기존 process_query 함수 - 새로운 스마트 버전 호출"""
    return await process_query_smart(query, text_placeholder, tool_placeholder, timeout_seconds)

async def initialize_basic_model():
    """MCP 도구 없이 기본 모델만 초기화합니다."""
    try:
        selected_model = st.session_state.selected_model

        model = ChatOpenAI(
            model=selected_model,
            temperature=0.1,
            max_tokens=4096,
        )

        st.session_state.basic_model = model
        st.session_state.basic_model_initialized = True
        return True
    except Exception as e:
        st.error(f"❌ 기본 모델 초기화 중 오류: {str(e)}")
        return False

class ToolManager:
    """도구를 관리하는 간단한 클래스"""
    
    @staticmethod
    async def get_tools_with_retry(mcp_config, max_retries=2):
        """재시도 로직이 포함된 도구 가져오기"""
        for attempt in range(max_retries):
            try:
                async with connect_all_mcp_servers(mcp_config) as (tools, server_sessions):
                    return tools
            except Exception as e:
                if attempt == max_retries - 1:  # 마지막 시도
                    raise e
                await asyncio.sleep(1)  # 1초 대기 후 재시도
        return []

async def initialize_session(mcp_config=None):
    """MCP 세션과 에이전트를 초기화합니다."""
    with st.spinner("🔄 AI 에이전트 초기화 중..."):
        # 기존 상태 정리
        st.session_state.session_initialized = False
        st.session_state.agent = None

        if mcp_config is None:
            mcp_config = load_config_from_json()

        try:
            # 연결할 서버들을 미리 연결 시도 상태로 설정
            print("🔄 MCP 서버 연결 시작...")
            for server_name in mcp_config.keys():
                update_server_status(server_name, "connecting", "서버 연결 시도 중...")
            
            # 간단한 방식으로 도구만 가져오기
            all_tools = await ToolManager.get_tools_with_retry(mcp_config)
            
            if not all_tools:
                # 모든 서버 연결 실패로 표시
                for server_name in mcp_config.keys():
                    update_server_status(server_name, "failed", "도구 로드 실패 - 연결된 도구 없음")
                st.warning("⚠️ 연결된 도구가 없습니다. 기본 모델만 사용됩니다.")
                return False
            
            st.session_state.tool_count = len(all_tools)
            st.session_state.available_tools = all_tools
            st.session_state.mcp_config = mcp_config

            # OpenAI 모델 초기화
            selected_model = st.session_state.selected_model

            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=4096,
            )
            translator_model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=4096,
            )

            # 모델들을 세션에 저장
            st.session_state.main_model = model
            st.session_state.translator_model = translator_model

            # LLM 분류기 초기화
            await WorkflowClassifier.initialize_llm_classifier()
            
            # 기본 워크플로우 타입은 일반적으로 설정 (동적 선택 사용)
            st.session_state.default_workflow_type = "general"
            
            print(f"🧠 LLM 기반 워크플로우 분류기 초기화 완료")
            print(f"🎯 기본 워크플로우 타입: general (질문별로 동적 선택)")
            
            # 기본 에이전트는 general로 설정 (실제로는 동적 선택)
            agent = WorkflowFactory.create_general_workflow(
                all_tools, model, translator_model, mcp_config
            ).compile(checkpointer=MemorySaver())

            st.session_state.agent = agent
            st.session_state.session_initialized = True
            
            print(f"✅ 에이전트 초기화 완료 - 총 {len(all_tools)}개 도구 연결됨")
            return True
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # 연결 실패한 서버들을 실패 상태로 표시
            for server_name in mcp_config.keys():
                current_status = st.session_state.server_status.get(server_name, {}).get("status", "unknown")
                if current_status != "connected":  # 이미 연결된 서버가 아니라면 실패로 표시
                    update_server_status(server_name, "failed", f"초기화 중 오류: {str(e)}")
            
            st.error(f"❌ MCP 서버 연결 중 오류: {str(e)}")
            st.error(f"상세 오류: {error_details}")
            return False

# --- 세션 상태 초기화 ---
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_connections = None  # 실제 MCP 연결 객체들 저장
    st.session_state.timeout_seconds = 120
    st.session_state.selected_model = "gpt-4o"
    st.session_state.recursion_limit = 100
    st.session_state.tool_count = 0
    st.session_state.basic_model = None
    st.session_state.basic_model_initialized = False
    st.session_state.server_status = {}  # 서버별 연결 상태 관리
    st.session_state.connection_logs = []  # 연결 로그
    st.session_state.conversation_memory_turns = 8  # 기억할 최대 대화 턴 수
    
    # 새로운 워크플로우 관리 변수들
    st.session_state.available_tools = []  # 사용 가능한 도구 목록
    st.session_state.main_model = None  # 메인 모델
    st.session_state.translator_model = None  # 번역 모델
    st.session_state.mcp_config = None  # MCP 설정
    st.session_state.default_workflow_type = "general"  # 기본 워크플로우 타입

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

if "pending_mcp_config" not in st.session_state:
    st.session_state.pending_mcp_config = load_config_from_json()

# 기본 모델 자동 초기화
if not st.session_state.basic_model_initialized:
    st.session_state.event_loop.run_until_complete(initialize_basic_model())

# --- 메인 UI ---
st.title("🤖 AI 에이전트")
st.markdown("✨ 다양한 도구를 활용하여 질문에 답변합니다. 의료 논문 검색 등 다양한 작업을 수행할 수 있습니다.")

# --- 사이드바 구성 ---
with st.sidebar:
    st.subheader("⚙️ 시스템 설정")

    # 모델 선택 (OpenAI만)
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    
    if has_openai_key:
        available_models = ["gpt-4o", "gpt-4o-mini"]
    else:
        st.warning("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 추가해주세요.")
        available_models = ["gpt-4o"]

    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 사용할 OpenAI 모델 선택",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        ),
        help="OpenAI API 키가 필요합니다."
    )

    if (
        previous_model != st.session_state.selected_model
        and st.session_state.session_initialized
    ):
        st.warning("⚠️ 모델이 변경되었습니다. '설정 적용하기' 버튼을 눌러 변경사항을 적용하세요.")

    # 타임아웃 설정
    st.session_state.timeout_seconds = st.slider(
        "⏱️ 응답 생성 제한 시간(초)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ 재귀 호출 제한(횟수)",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
    )

    # 대화 메모리 설정
    st.session_state.conversation_memory_turns = st.slider(
        "🧠 멀티턴 메모리 대화 수",
        min_value=3,
        max_value=20,
        value=st.session_state.conversation_memory_turns,
        step=1,
        help="이전 대화를 몇 턴까지 기억할지 설정합니다. 값이 클수록 더 많은 토큰을 사용합니다."
    )

    st.divider()

    # 도구 설정 섹션
    st.subheader("🔧 의료 도구 설정")

    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    with st.expander("🧰 MCP 도구 추가", expanded=st.session_state.mcp_tools_expander):
        st.markdown("""
        [설정 가이드](https://teddylee777.notion.site/MCP-1d324f35d12980c8b018e12afdf545a1?pvs=4)
        
        ⚠️ **중요**: JSON을 반드시 중괄호(`{}`)로 감싸야 합니다.
        """)

        example_json = {
            "pubmed": {
                "command": "cmd",
                "args": [
                    "/c",
                    "npx",
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@JackKuo666/pubmed-mcp-server",
                    "--key",
                    "a0fde5b8-88e9-46d3-ab62-ad5096bd7d4b",
                ],
                "transport": "stdio",
            }
        }

        default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

        new_tool_json = st.text_area(
            "의료 도구 JSON",
            default_text,
            height=250,
        )

        if st.button("도구 추가", type="primary", use_container_width=True):
            try:
                if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                    st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                else:
                    parsed_tool = json.loads(new_tool_json)

                    if "mcpServers" in parsed_tool:
                        parsed_tool = parsed_tool["mcpServers"]

                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        if "transport" not in tool_config:
                            tool_config["transport"] = "stdio"

                        if "command" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'command' 필드가 필요합니다.")
                        elif "args" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다.")
                        elif not isinstance(tool_config["args"], list):
                            st.error(f"'{tool_name}' 도구의 'args' 필드는 배열([]) 형식이어야 합니다.")
                        else:
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(f"{success_tools[0]} 도구가 추가되었습니다.")
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다.")
                        st.session_state.mcp_tools_expander = False
                        st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON 파싱 에러: {e}")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    # 등록된 도구 목록
    with st.expander("📋 등록된 의료 도구 목록", expanded=True):
        pending_config = st.session_state.pending_mcp_config
        
        if not pending_config:
            st.info("아직 등록된 도구가 없습니다. 위에서 도구를 추가해보세요.")
        else:
            for tool_name in list(pending_config.keys()):
                col1, col2 = st.columns([8, 2])
                
                # 기본 도구 이름 표시
                col1.markdown(f"- **{tool_name}**")
                
                # 실제 연결된 도구에서 description 가져오기
                if st.session_state.session_initialized:
                    tool_desc = get_tool_description_from_available_tools(tool_name)
                    if tool_desc:
                        col1.markdown(f"  <small style='color: #666;'>{tool_desc}</small>", unsafe_allow_html=True)
                else:
                    col1.markdown(f"  <small style='color: #999;'>연결 후 설명을 확인할 수 있습니다</small>", unsafe_allow_html=True)
                
                # 삭제 버튼
                if col2.button("삭제", key=f"delete_{tool_name}"):
                    del st.session_state.pending_mcp_config[tool_name]
                    st.success(f"{tool_name} 도구가 삭제되었습니다.")
                    st.rerun()

    st.divider()

    # MCP 서버 연결 상태
    st.subheader("🔗 MCP 서버 연결 상태")
    
    if st.session_state.server_status:
        for server_name, status_info in st.session_state.server_status.items():
            status = status_info["status"]
            tools_count = status_info["tools_count"]
            error_message = status_info["error_message"]
            
            # 상태별 아이콘과 색상
            if status == "connected":
                status_icon = "✅"
                status_color = "success"
                status_text = f"연결됨 ({tools_count}개 도구)"
            elif status == "connecting":
                status_icon = "🔄"
                status_color = "info"
                status_text = "연결 중..."
            else:  # failed
                status_icon = "❌"
                status_color = "error"
                status_text = "연결 실패"
            
            with st.container():
                st.markdown(f"{status_icon} **{server_name}**: {status_text}")
                

                
                if error_message and status != "connected":
                    with st.expander(f"❗ {server_name} 오류 정보", expanded=False):
                        st.error(error_message)
                        
                        # 일반적인 해결방안 제시
                        st.markdown("**해결방안:**")
                        if "타임아웃" in error_message:
                            st.markdown("**📡 네트워크 문제일 가능성:**")
                            st.markdown("- 인터넷 연결 상태를 확인해주세요")
                            st.markdown("- VPN 사용 중이면 잠시 끄고 시도해주세요")
                            st.markdown("- 회사/학교 네트워크에서 npm 패키지 다운로드가 차단될 수 있습니다")
                            st.markdown("- 방화벽이나 프록시 설정을 확인해주세요")
                            st.markdown("")
                            st.markdown("**🔧 Node.js 환경 확인:**")
                            st.markdown("- 터미널에서 `node --version` 실행")
                            st.markdown("- 터미널에서 `npm --version` 실행")
                            st.markdown("- Node.js가 없다면 https://nodejs.org 에서 설치")
                        elif "npm" in error_message or "npx" in error_message:
                            st.markdown("- Node.js가 설치되어 있는지 확인해주세요")
                            st.markdown("- `npm install -g @smithery/cli` 실행해보세요")
                            st.markdown("- `npx -y @smithery/cli@latest run @JackKuo666/pubmed-mcp-server --key a0fde5b8-88e9-46d3-ab62-ad5096bd7d4b` 직접 실행해보세요")
                        else:
                            st.markdown("- 네트워크 연결을 확인해주세요")
                            st.markdown("- 잠시 후 다시 시도해주세요")
                            st.markdown("- PubMed API 서버의 일시적 문제일 수 있습니다")
    else:
        st.info("아직 서버 연결을 시도하지 않았습니다.")
    
    # 연결 로그 표시
    if st.session_state.connection_logs:
        with st.expander("📋 연결 로그", expanded=False):
            for log in st.session_state.connection_logs[-10:]:  # 최근 10개만 표시
                st.text(log)

    st.divider()

    # 시스템 정보
    st.subheader("📊 시스템 정보")
    registered_tools = len(st.session_state.pending_mcp_config)
    connected_tools = st.session_state.get('tool_count', 0)
    
    st.write(f"🛠️ 등록된 도구 수: {registered_tools}개")
    if st.session_state.session_initialized:
        st.write(f"🔗 연결된 도구 수: {connected_tools}개")
        
        # 전체 연결된 도구 목록 보기 버튼
        if connected_tools > 0:
            if st.button("📋 전체 도구 목록", key="all_tools_button", help="연결된 모든 도구의 설명 보기"):
                st.session_state.show_all_tools = not st.session_state.get('show_all_tools', False)
            
            # 전체 도구 목록 표시 (서버별 구분)
            if st.session_state.get('show_all_tools', False):
                with st.expander("🔧 연결된 모든 도구들 (서버별)", expanded=True):
                    if st.session_state.get('available_tools'):
                        # 서버별로 도구들을 그룹화
                        server_tools_map = {}
                        
                        for tool in st.session_state.available_tools:
                            # 도구 이름으로 서버 추정
                            tool_server = "기타"
                            tool_name_lower = tool.name.lower()
                            
                            if "pubmed" in tool_name_lower or "medline" in tool_name_lower:
                                tool_server = "📚 PubMed"
                            elif any(keyword in tool_name_lower for keyword in ["search_", "webkr", "news", "blog", "shop", "image", "kin", "book", "encyc", "academic", "local", "cafe", "datalab"]):
                                tool_server = "🔍 웹검색"
                            elif "weather" in tool_name_lower or "climate" in tool_name_lower:
                                tool_server = "🌤️ 날씨"
                            elif "time" in tool_name_lower or "clock" in tool_name_lower:
                                tool_server = "⏰ 시간"
                            elif "calc" in tool_name_lower or "math" in tool_name_lower:
                                tool_server = "🧮 계산기"
                            else:
                                tool_server = "🔧 기타"
                            
                            if tool_server not in server_tools_map:
                                server_tools_map[tool_server] = []
                            server_tools_map[tool_server].append(tool)
                        
                        # 서버별로 도구 표시
                        for server, tools in server_tools_map.items():
                            st.markdown(f"## {server} ({len(tools)}개 도구)")
                            
                            tools_with_desc = []
                            tools_without_desc = []
                            
                            for tool in tools:
                                if hasattr(tool, 'description') and tool.description and tool.description.strip():
                                    tools_with_desc.append(tool)
                                else:
                                    tools_without_desc.append(tool)
                            
                            # description이 있는 도구들
                            if tools_with_desc:
                                st.markdown("**📝 설명이 있는 도구들:**")
                                for i, tool in enumerate(tools_with_desc, 1):
                                    with st.container():
                                        st.markdown(f"**{i}. {tool.name}**")
                                        st.markdown(f"📝 {tool.description}")
                                        
                                        # 매개변수 표시
                                        if hasattr(tool, 'args_schema') and tool.args_schema:
                                            try:
                                                if hasattr(tool.args_schema, 'model_fields'):
                                                    fields = list(tool.args_schema.model_fields.keys())
                                                    if fields:
                                                        st.markdown(f"⚙️ **매개변수:** {', '.join(fields[:5])}{'...' if len(fields) > 5 else ''}")
                                            except:
                                                pass
                                        st.markdown("---")
                            
                            # description이 없는 도구들
                            if tools_without_desc:
                                st.markdown(f"**🔧 기타 도구들:** {len(tools_without_desc)}개 (설명 없음)")
                                tool_names = [tool.name for tool in tools_without_desc]
                                st.markdown(f"📋 {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}")
                            
                            st.markdown("")  # 서버 간 간격
                            st.markdown("---")  # 서버 간 구분선
                        
                        if not server_tools_map:
                            st.info("연결된 도구가 없습니다.")
                    else:
                        st.error("도구 정보를 불러올 수 없습니다.")
    else:
        st.write(f"🔗 연결된 도구 수: 초기화 필요")
    st.write(f"🧠 현재 모델: {st.session_state.selected_model}")
    
    # 대화 상태 정보
    conversation_turns = len([msg for msg in st.session_state.history if msg["role"] in ["user", "assistant"]]) // 2
    st.write(f"💬 현재 대화 턴: {conversation_turns}턴")
    st.write(f"🧠 메모리 설정: 최대 {st.session_state.conversation_memory_turns}턴 기억")

    # 설정 적용하기 버튼
    if st.button("설정 적용하기", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 의료 에이전트를 초기화하고 있습니다...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 0단계: 연결 상태 초기화 (사이드바 갱신을 위해)
            status_text.text("연결 상태 초기화 중...")
            st.session_state.server_status = {}  # 서버 상태 완전 초기화
            st.session_state.connection_logs = []  # 연결 로그 초기화
            progress_bar.progress(10)

            # 1단계: 설정 저장
            status_text.text("설정 파일 저장 중...")
            save_result = save_config_to_json(st.session_state.pending_mcp_config)
            progress_bar.progress(25)

            # 2단계: 기존 연결 정리
            status_text.text("기존 연결 정리 중...")
            st.session_state.session_initialized = False
            st.session_state.agent = None
            st.session_state.available_tools = []  # 기존 도구 목록 정리
            st.session_state.tool_count = 0
            progress_bar.progress(40)

            # 3단계: 각 서버를 연결 대기 상태로 설정
            status_text.text("서버 연결 준비 중...")
            for server_name in st.session_state.pending_mcp_config.keys():
                update_server_status(server_name, "connecting", "연결 준비 중...")
            progress_bar.progress(50)

            # 4단계: MCP 서버 연결 시도
            status_text.text("MCP 서버 연결 중...")
            progress_bar.progress(60)
            
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )
            progress_bar.progress(100)

            if success:
                status_text.text("초기화 완료!")
                st.success("✅ 의료 논문 검색 에이전트가 준비되었습니다!")
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                status_text.text("초기화 실패")
                st.error("❌ 에이전트 초기화에 실패했습니다. 위의 연결 상태를 확인해주세요.")

        st.rerun()
    
    # 재연결 버튼 (연결 실패한 서버가 있을 때만 표시)
    failed_servers = [name for name, status in st.session_state.server_status.items() 
                     if status["status"] == "failed"]
    
    if failed_servers:
        if st.button("🔄 실패한 서버 재연결", key="reconnect_button", use_container_width=True):
            with st.spinner("🔄 실패한 서버들 재연결 중..."):
                # 실패한 서버들의 상태를 연결 시도로 변경
                for server_name in failed_servers:
                    update_server_status(server_name, "connecting", "재연결 시도 중...")
                
                # 연결 로그 추가
                st.session_state.connection_logs.append(f"[재연결] {len(failed_servers)}개 서버 재연결 시도")
                
                # 재연결 시도
                success = st.session_state.event_loop.run_until_complete(
                    initialize_session(st.session_state.pending_mcp_config)
                )
                
                if success:
                    st.success("✅ 재연결이 완료되었습니다!")
                else:
                    st.warning("⚠️ 일부 서버 재연결에 실패했습니다.")
            
            st.rerun()

    st.divider()

    # 작업 버튼들
    st.subheader("🔄 작업")

    if st.button("🗑️ 대화 초기화", use_container_width=True, type="primary"):
        # 대화 기록 및 메모리 완전 초기화
        st.session_state.thread_id = random_uuid()
        st.session_state.history = []
        
        # LangGraph 메모리도 새로운 thread_id로 초기화됨
        st.success("✅ 대화 기록과 메모리가 모두 초기화되었습니다.")
        st.rerun()
        
    # 대화 요약 버튼 (대화가 많을 때만 표시)
    conversation_turns = len([msg for msg in st.session_state.history if msg["role"] in ["user", "assistant"]]) // 2
    if conversation_turns > st.session_state.conversation_memory_turns:
        if st.button("📝 오래된 대화 요약", use_container_width=True):
            st.info("💡 현재 설정된 메모리 턴 수를 초과한 대화는 자동으로 관리됩니다.")
            st.info(f"🔄 현재 {conversation_turns}턴 중 최근 {st.session_state.conversation_memory_turns}턴만 기억합니다.")

# --- 메인 컨텐츠 ---
# 대화 상태 표시
conversation_turns = len([msg for msg in st.session_state.history if msg["role"] in ["user", "assistant"]]) // 2

if not st.session_state.session_initialized:
    if st.session_state.basic_model_initialized:
        if conversation_turns > 0:
            st.info(f"💬 기본 모델 대화 진행 중 ({conversation_turns}턴) - 이전 대화를 기억합니다!")
        else:
            st.info("💬 기본 모델이 준비되었습니다! 일반적인 질문은 지금 바로 할 수 있습니다.")
        st.warning("🔧 의료 논문 검색 기능을 사용하려면 사이드바의 '설정 적용하기' 버튼을 클릭해주세요.")
    else:
        st.info("🏥 의료 논문 검색 에이전트가 초기화되지 않았습니다. 왼쪽 사이드바의 '설정 적용하기' 버튼을 클릭하여 초기화해주세요.")
else:
    workflow_type = st.session_state.get('default_workflow_type', 'general')
    if conversation_turns > 0:
        st.success(f"✅ AI 에이전트 준비 완료! 현재 {conversation_turns}턴 대화 진행 중 (기본: {workflow_type} 워크플로우)")
        st.info("🧠 이전 대화 내용을 기억하며 질문 유형에 따라 적절한 워크플로우를 자동 선택합니다.")
    else:
        st.success(f"✅ AI 에이전트가 준비되었습니다! (기본: {workflow_type} 워크플로우) 한글로 질문해보세요.")

# 대화 기록 출력
print_message()

# 사용자 입력 처리
# 대화 상태에 따른 플레이스홀더 텍스트 설정
if conversation_turns > 0:
    if st.session_state.session_initialized:
        placeholder_text = f"💬 추가 질문을 입력하세요 (현재 {conversation_turns}턴, 이전 대화 기억함)"
    else:
        placeholder_text = f"💬 추가 질문을 입력하세요 (현재 {conversation_turns}턴, 기본 모델)"
else:
    placeholder_text = "💬 의료 관련 질문을 입력하세요 (한글 가능)"

user_query = st.chat_input(placeholder_text)
if user_query:
    st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
    
    if st.session_state.session_initialized:
        # 완전한 의료 논문 검색 에이전트 사용
        with st.chat_message("assistant", avatar="🏥"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            if final_tool.strip():
                st.session_state.history.append({"role": "assistant_tool", "content": final_tool})
            st.rerun()
    
    elif st.session_state.basic_model_initialized:
        # 기본 모델만 사용
        with st.chat_message("assistant", avatar="🤖"):
            text_placeholder = st.empty()
            
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_basic_query(
                        user_query,
                        text_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            
            # 멀티턴 대화 정보 표시
            new_turn_count = len([msg for msg in st.session_state.history if msg["role"] in ["user", "assistant"]]) // 2
            if new_turn_count > 1:
                st.info(f"ℹ️ 기본 모델로 답변 ({new_turn_count}턴째, 이전 대화 기억함). 의료 논문 검색을 원하시면 '설정 적용하기'를 눌러주세요.")
            else:
                st.info("ℹ️ 기본 모델로 답변했습니다. 의료 논문 검색을 원하시면 '설정 적용하기'를 눌러주세요.")
            st.rerun()
    
    else:
        st.warning("⚠️ 모델이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")
