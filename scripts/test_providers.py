#!/usr/bin/env python3
"""
Test script for different LLM providers using litellm unified interface.
Tests Google/Gemini, OpenAI, and Amazon Bedrock endpoints.

Make sure to set the appropriate environment variables before running:
- GEMINI_API_KEY for Google/Gemini
- OPENAI_API_KEY for OpenAI  
- AWS credentials for Bedrock (via aws sso login or environment variables)
"""

import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_provider(model: str, provider_name: str, env_var: Optional[str] = None) -> Optional[str]:
    """Test a specific LLM provider using litellm."""
    try:
        from litellm import completion
        
        # Check if required environment variable is set
        if env_var and not os.getenv(env_var):
            logger.warning(f"{env_var} not set, skipping {provider_name} test")
            return None
            
        logger.info(f"Testing {provider_name} endpoint (model: {model})...")
        
        messages = [{"role": "user", "content": "In one sentence, what is zero-shot coordination?"}]
        
        # Make the completion call
        response = completion(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        
        # Extract content from response dynamically to avoid type issues
        try:
            choices = getattr(response, 'choices', None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    content = getattr(message, 'content', None)
                    result = str(content or "").strip()
                else:
                    result = str(choices[0]).strip()
            else:
                result = str(response).strip()
        except Exception:
            result = str(response).strip()
        
        logger.info(f"✅ {provider_name} response: {result}")
        return result
        
    except ImportError:
        logger.error("❌ litellm not installed. Run: pip install litellm")
        return None
    except Exception as e:
        logger.error(f"❌ {provider_name} test failed: {e}")
        return None


def main():
    """Run tests for all LLM providers using litellm."""
    print("🚀 Testing LLM Providers with litellm\n" + "="*50)
    
    # Define provider configurations
    providers = [
        {
            "name": "Google Gemini",
            "model": "gemini/gemini-2.5-flash", 
            "env_var": "GEMINI_API_KEY"
        },
        {
            "name": "OpenAI",
            "model": "gpt-4o-mini",
            "env_var": "OPENAI_API_KEY"
        },
        {
            "name": "Amazon Bedrock",
            "model": "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
            "env_var": None  # Uses AWS credentials
        }
    ]
    
    results = {}
    
    # Test each provider
    for provider in providers:
        results[provider["name"]] = test_provider(
            model=provider["model"],
            provider_name=provider["name"],
            env_var=provider["env_var"]
        )
        print()

    
    # Summary
    print("📊 Test Summary:")
    print("-" * 40)
    for provider_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED/SKIPPED"
        print(f"{provider_name:>15}: {status}")
    
    successful_tests = sum(1 for result in results.values() if result is not None)
    total_tests = len(results)
    print(f"\nSuccessful tests: {successful_tests}/{total_tests}")
    
    if successful_tests == 0:
        print("\n💡 Setup instructions:")
        print("   - For Gemini: Set GEMINI_API_KEY environment variable")
        print("   - For OpenAI: Set OPENAI_API_KEY environment variable")  
        print("   - For Bedrock: Run 'aws sso login' or set AWS credentials")


if __name__ == "__main__":
    main()