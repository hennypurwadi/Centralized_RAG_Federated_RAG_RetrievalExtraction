# 🚀 HuggingFace Spaces Deployment Guide

## Step-by-Step Deployment Instructions

### 1. Create a New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `federated-rag-chatbot` (or your preferred name)
   - **License**: Apache 2.0 (recommended)
   - **SDK**: Select "Gradio"
   - **Hardware**: CPU Basic (free tier) or upgrade as needed
   - **Visibility**: Public or Private (your choice)

### 2. Upload Files

Upload all the following files to your Space:

**Required Files:**
- `app.py` - Main entry point
- `True_Federated_RAG_Chatbot_g.py` - Core implementation
- `requirements.txt` - Dependencies
- `msc_ai_hullonline_short.txt` - Hull University data
- `msc_ai_keeleonline.txt` - Keele University data
- `README.md` - Documentation

### 3. Configuration

The Space will automatically:
- Install dependencies from `requirements.txt`
- Start the application using `app.py`
- Make the interface available at your Space URL

### 4. Environment Variables (Optional)

If you want to pre-configure an OpenAI API key:
1. Go to your Space settings
2. Add a new secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key

### 5. Hardware Requirements

**Minimum (Free Tier):**
- CPU Basic: Sufficient for basic usage
- Memory: 16GB RAM included

**Recommended (Paid Tier):**
- CPU Upgrade: For better performance with multiple users
- Persistent Storage: If you plan to add more data files

### 6. Testing Your Deployment

1. Wait for the Space to build (usually 2-5 minutes)
2. Open your Space URL
3. Test the interface:
   - Enter an OpenAI API key
   - Click "Initialize Federated System"
   - Ask a sample question
   - Verify the federated responses work

### 7. Customization Options

**Data Files:**
- Replace `msc_ai_hullonline_short.txt` and `msc_ai_keeleonline.txt` with your own data
- Update file paths in the interface if needed

**Styling:**
- Modify the CSS in `True_Federated_RAG_Chatbot_g.py` to change appearance
- Update colors, fonts, or layout as desired

**Models:**
- Change default models in the dropdown options
- Add support for other LLM providers if needed

### 8. Monitoring and Maintenance

**Logs:**
- Check Space logs for any errors
- Monitor usage and performance

**Updates:**
- Upload new versions of files to update the Space
- The Space will automatically restart with changes

**Scaling:**
- Upgrade hardware if you experience performance issues
- Consider persistent storage for larger datasets

### 9. Troubleshooting

**Common Issues:**

1. **Build Failures:**
   - Check `requirements.txt` for correct package versions
   - Ensure all files are uploaded correctly

2. **Runtime Errors:**
   - Verify OpenAI API key is valid
   - Check data file paths are correct

3. **Performance Issues:**
   - Consider upgrading to a paid hardware tier
   - Optimize chunk sizes and retrieval parameters

4. **Memory Issues:**
   - Reduce chunk sizes in the configuration
   - Limit the number of documents retrieved

### 10. Example Space Configuration

**Space Settings:**
```yaml
title: "🔗 Federated RAG Chatbot"
emoji: "🔗"
colorFrom: "blue"
colorTo: "purple"
sdk: "gradio"
sdk_version: "5.14.0"
app_file: "app.py"
pinned: false
```

### 11. Security Considerations

- Never hardcode API keys in your code
- Use HuggingFace Secrets for sensitive information
- Consider rate limiting for public Spaces
- Monitor usage to prevent abuse

### 12. Sharing Your Space

Once deployed, you can:
- Share the Space URL with others
- Embed the Space in websites using HuggingFace's embed feature
- Create a duplicate for others to customize

---

**Need Help?**
- Check HuggingFace Spaces documentation
- Visit the HuggingFace community forums
- Review the example Space at the provided URL

