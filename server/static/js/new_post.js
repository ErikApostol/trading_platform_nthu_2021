function post_data(url, data) {
    return fetch(url, {
      body: JSON.stringify(data),
      cache: 'no-cache',
      credentials: 'same-origin',
      headers: {
        'user-agent': 'Mozilla/4.0 MDN Example',
        'content-type': 'application/json'
      },
      method: 'POST',
      mode: 'cors',
      referrer: 'no-referrer',
    })
    .then(response => response.json())
}

function post_comment() {
    var data = {
        title: document.getElementById("title").value, 
        video_id: parseInt(document.getElementById("video_id").value), 
        comment: document.getElementById("comment").value, 
        tag: "test"
    };

    post_data("/post", data);
}

async function insert_data() {
    const response = await fetch("/get_result");
    const data = await response.json();

    var select = document.getElementById("video_id");

    console.log(data);
    
    for(var i=0; i<data["count"]; i++) {
        var tmp_data = data["content"][i];
        
        var btn = document.createElement('option');
        btn.value = tmp_data["video_id"];
        btn.text = tmp_data["video_id"];
        select.appendChild(btn);
    }
}

insert_data();
