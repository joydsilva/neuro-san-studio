#!/usr/bin/env python3
"""
Standalone MCP Quote Generation Client Example
Demonstrates how to use langchain-mcp-adapters to connect to the quote MCP server
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient


async def test_mcp_quote_client():
    """Test the MCP quote generation client directly."""
    
    # Configure MCP client connection
    client = MultiServerMCPClient(
        {
            "create-quote": {
                "transport": "stdio",
                "command": "node",
                "args": ["c:\\POC\\quote_mcp_server_simple\\index.js"],
                "env": {}
            }
        }
    )
    
    try:
        # Get available tools from MCP server
        print("🔌 Connecting to MCP server...")
        tools = await client.get_tools()
        
        if not tools:
            print("❌ No tools available from MCP server")
            return
        
        print(f"✅ Connected! Found {len(tools)} tool(s):")
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.name}: {tool.description}")
        
        # Prepare sample quote request
        print("\n📝 Preparing quote request...")
        sample_request = {
            "quote": {
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "expiration_date": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
                "quote_status": "Active",
                "locations": [
                    {
                        "address": "123 Business St, New York, NY 10001",
                        "property_type": "Restaurant",
                        "square_footage": 5000
                    }
                ],
                "lines": [
                    {
                        "line_type": "General Liability",
                        "policy_limit": 2000000,
                        "deductible": 1000,
                        "risks": [
                            {
                                "risk_type": "General Business Risk",
                                "location_reference": "LOC-001",
                                "risk_description": "General Liability risk for Restaurant operation",
                                "coverages": [
                                    {
                                        "coverage_type": "General Liability",
                                        "coverage_limit": 2000000
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        
        print("📤 Sending request to MCP server...")
        print(f"Request: {json.dumps(sample_request, indent=2)}")
        
        # Call the quote creation tool (second tool, index 1)
        quote_tool = None
        for tool in tools:
            if "quote" in tool.name.lower():
                quote_tool = tool
                break
        
        if not quote_tool:
            print("❌ Quote creation tool not found")
            return
            
        print(f"📞 Using tool: {quote_tool.name}")
        result = await quote_tool.ainvoke(sample_request)
        
        print("\n📥 Response received:")
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up connection if available
        if hasattr(client, 'close'):
            await client.close()


async def test_simple_parameters():
    """Test with simple parameter format that might be expected by MCP server."""
    
    client = MultiServerMCPClient(
        {
            "create-quote": {
                "transport": "stdio", 
                "command": "node",
                "args": ["c:\\POC\\quote_mcp_server_simple\\index.js"],
                "env": {}
            }
        }
    )
    
    try:
        tools = await client.get_tools()
        if not tools:
            print("❌ No tools available")
            return
        
        # Try with simplified parameters
        simple_request = {
            "business_type": "Restaurant",
            "coverage_type": "General Liability", 
            "coverage_limit": "2000000",
            "location": "123 Business St, New York, NY 10001",
            "square_footage": "5000",
            "deductible": "1000"
        }
        
        print("\n🔄 Testing with simple parameters...")
        print(f"Request: {json.dumps(simple_request, indent=2)}")
        
        # Find the quote creation tool
        quote_tool = None
        for tool in tools:
            if "quote" in tool.name.lower():
                quote_tool = tool
                break
        
        if not quote_tool:
            print("❌ Quote creation tool not found")
            return
            
        print(f"📞 Using tool: {quote_tool.name}")
        result = await quote_tool.ainvoke(simple_request)
        print(f"✅ Result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error with simple parameters: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Starting MCP Quote Client Test")
    print("=" * 50)
    
    # Test both complex and simple parameter formats
    asyncio.run(test_mcp_quote_client())
    asyncio.run(test_simple_parameters())
    
    print("\n✅ Test completed!")