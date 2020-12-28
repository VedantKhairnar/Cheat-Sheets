import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './style.scss';

function AboutUs() {

    const [data, setData] = useState([]);

    useEffect(() => {
        axios.get('https://api.github.com/repos/VedantKhairnar/Cheat-Sheets/contributors')
            .then(response => {
                console.log(response.data);
                setData(response.data);
            })
    }, [])

    return (
        <div className="AboutUs">
            <div className="AboutUs-title">
                About Us
            </div>
            <div className="AboutUs-body">
                List of all science cheat sheets.
            </div>
            <div className="AboutUs-title">
                Thanks to all our Contributors
            </div>
            <div className="Contributors">
                {
                    data && data.map(d => (
                        <a className="Contributors-box" key={d.login} href={d.html_url} target="_blank" rel="noreferrer">
                            <div className="name">{d.login}</div>
                            <img src={d.avatar_url} alt="GithubProfile" />
                            <div className="number">Contributions: {d.contributions}</div>
                        </a>
                    ))
                }
            </div>
        </div>
    )
}

export default AboutUs
